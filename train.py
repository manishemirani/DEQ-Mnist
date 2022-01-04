import jax
import optax
import numpy as np
import jax.numpy as jnp
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
from functools import partial
from typing import Tuple, Callable
from fixed_point_solvers import anderson_solver
from tensorflow.keras.datasets import mnist
from tqdm import tqdm

# define hparams
CNN_CHANNELS = 154
CHANNELS = 28
NUM_GROUP = 7
NUM_GROUPS = 7
LEARNING_RATE = 0.001
BATCH_SIZE = 100
EPOCHS = 50


class CNN(nn.Module):
    channels: int
    output_channels: int
    num_groups: int
    kernel_size: Tuple = (3, 3)
    use_bias: bool = False

    def setup(self):
        self.conv1 = nn.Conv(features=self.channels, kernel_size=self.kernel_size,
                             kernel_init=nn.initializers.normal(),
                             use_bias=self.use_bias)
        self.conv2 = nn.Conv(features=self.output_channels, kernel_size=self.kernel_size,
                             kernel_init=nn.initializers.normal(),
                             use_bias=self.use_bias)
        self.group_norm1 = nn.GroupNorm(self.num_groups)
        self.group_norm2 = nn.GroupNorm(self.num_groups)
        self.group_norm3 = nn.GroupNorm(self.num_groups)

    @nn.compact
    def __call__(self, x, z):
        y = self.group_norm1(nn.relu(self.conv1(z)))
        return self.group_norm3(nn.relu(z + self.group_norm2(x + self.conv2(y))))


# Define Deep Equilibrium model
class DEQ(nn.Module):
    cnn_channels: int
    channels: int
    num_groups: int
    solver: Callable
    f: nn.Module
    classes: int = 10

    def setup(self):
        self.model = self.f(channels=self.cnn_channels, output_channels=self.channels,
                            num_groups=self.num_groups)
        self.conv1 = nn.Conv(features=self.channels, kernel_size=(3, 3))
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dense = nn.Dense(self.classes)

    @nn.compact
    def __call__(self, x, cnn_params):
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.norm1(x)
        x = fixed_point_layer(self.solver, self.model, cnn_params, x)
        x = self.norm2(x)
        x = x.reshape((x.shape[0], -1))
        x = self.dense(x)
        return x


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def fixed_point_layer(solver, f: nn.Module, params, x):
    z_star = solver(lambda z: f.apply({'params': params}, x, z), z_init=jnp.zeros_like(x))
    return z_star


def fixed_point_fwd(solver, f, params, x):
    z_star = fixed_point_layer(solver, f, params, x)
    return z_star, (params, x, z_star)


# Custom backward pass for fixed point layer
def fixed_point_bwd(solver, f: nn.Module, res, z_star_bar):
    params, x, z_star = res
    _, vjp_a = jax.vjp(lambda params, x: f.apply({'params': params}, x, z_star), params, x)
    _, vjp_z = jax.vjp(lambda z: f.apply({'params': params}, x, z), z_star)
    return vjp_a(solver(lambda u: vjp_z(u)[0] + z_star_bar,
                        z_init=jnp.zeros_like(z_star)))


fixed_point_layer.defvjp(fixed_point_fwd, fixed_point_bwd)

cnn = CNN(channels=CNN_CHANNELS,
          output_channels=CHANNELS,
          num_groups=NUM_GROUP)

deq = DEQ(cnn_channels=CNN_CHANNELS,
          channels=CHANNELS,
          num_groups=NUM_GROUP,
          solver=anderson_solver,
          f=CNN)


def create_state_for_CNN(rng, dummy_input, learning_rate):
    """Creates initial state for CNN model"""
    params = cnn.init(rng, dummy_input, dummy_input)['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=cnn.apply,
        params=params,
        tx=tx
    )


def create_state_for_deq(rng, dummy_input, learning_rate,
                         cnn_params):
    """Creates initial state for DEQ model"""
    params = deq.init(rng, dummy_input, cnn_params)['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=deq.apply,
        params=params,
        tx=tx
    )


def update_cnn_params(state, images):
    """Updating CNN model's parameters with applying gradients"""
    grads = jax.grad(lambda params: fixed_point_layer(anderson_solver,
                                                      cnn, params, images).sum())(state.params)
    return state.apply_gradients(grads=grads)


def update_deq_params(state, grads):
    """Updating DEQ model's parameters with applying gradients"""
    return state.apply_gradients(grads=grads)


def apply_model(deq_state, cnn_state, images, labels):
    """Apply gradients for DEQ model"""

    def loss_fn(deq_params):
        logits = deq.apply({'params': deq_params},
                           images,
                           cnn_state.params)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(deq_state.params)
    acc = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, acc


def train_epoch(deq_state, cnn_state, images, labels, batch_size, rng, epoch):
    """Train a single epoch"""
    step_per_epoch = len(images) // batch_size

    # Choosing random data for training
    perms = jax.random.permutation(rng, len(images))
    perms = perms[:step_per_epoch * batch_size]
    perms = perms.reshape((step_per_epoch, batch_size))
    epoch_loss = []
    epoch_acc = []
    with tqdm(perms) as tq:
        for perm in tq:
            tq.set_description(f"Epoch {epoch}")
            batch_images = images[perm, ...]
            batch_labels = labels[perm, ...]
            grads, loss, acc = apply_model(deq_state=deq_state,
                                           cnn_state=cnn_state,
                                           images=batch_images,
                                           labels=batch_labels)
            deq_state = update_deq_params(deq_state, grads)
            cnn_state = update_cnn_params(cnn_state, batch_images)
            tq.set_postfix(loss=loss, accuracy=acc * 100)
            epoch_loss.append(loss)
            epoch_acc.append(acc)

    train_loss = np.mean(epoch_loss)
    train_acc = np.mean(epoch_acc)
    return deq_state, cnn_state, train_loss, train_acc


def get_dataset():
    """Getting dataset and preprocess it"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = jnp.float32(x_train)
    x_test = jnp.float32(x_test)
    x_train /= 255.
    x_test /= 255.
    return x_train, y_train, x_test, y_test


def train(work_dir):
    train_images, train_labels, test_images, test_labels = get_dataset()
    rng = jax.random.PRNGKey(0)
    rng, init_rng_deq, init_rng_cnn = jax.random.split(rng, num=3)
    cnn_state = create_state_for_CNN(rng=init_rng_cnn,
                                     dummy_input=train_images[0][jnp.newaxis, ...],
                                     learning_rate=LEARNING_RATE)

    deq_state = create_state_for_deq(rng=init_rng_deq,
                                     dummy_input=train_images[0][jnp.newaxis, ...],
                                     cnn_params=cnn_state.params,
                                     learning_rate=LEARNING_RATE)

    # using tensorboard to check model's behaviour
    summary = tensorboard.SummaryWriter(work_dir)
    summary.hparams({'learning_rate': LEARNING_RATE,
                     'batch_size': BATCH_SIZE,
                     'num_epochs': EPOCHS})

    for epoch in range(1, EPOCHS + 1):
        rng, _, _ = jax.random.split(rng, num=3)
        deq_state, cnn_state, train_loss, train_acc = train_epoch(
            deq_state=deq_state,
            cnn_state=cnn_state,
            images=train_images,
            labels=train_labels,
            batch_size=BATCH_SIZE,
            epoch=epoch,
            rng=rng
        )  # train a single epoch
        _, test_loss, test_acc = apply_model(deq_state=deq_state,
                                             cnn_state=cnn_state,
                                             images=test_images,
                                             labels=test_labels)  # model testing
        print(f"Testing at epoch {epoch}")
        print("test loss: %.2f, test accuracy: %2.f" % (test_loss, test_acc * 100))

        summary.scalar('train_loss', train_loss, epoch)
        summary.scalar('train_accuracy', train_acc, epoch)
        summary.scalar('test_loss', test_loss, epoch)
        summary.scalar('test_accuracy', test_acc, epoch)

    summary.flush()


train(work_dir='./logs')