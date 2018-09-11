from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import all_aes
import tensorflow as tf
from lib import data, utils
import numpy as np
import ipdb

FLAGS = flags.FLAGS
flags.DEFINE_string('ae_dir', '', 'Folder containing AE to use for DA.')


def get_images_and_latents_and_labels(sess, ops, dataset, batches=None):
    batch = FLAGS.batch
    with tf.Graph().as_default():
        data_in = dataset.make_one_shot_iterator().get_next()
        with tf.Session() as sess_new:
            images_orig = []
            labels = []
            images_processed = []
            while True:
                try:
                    payload = sess_new.run(data_in)
                    images_orig.append(payload['x_orig'])
                    images_processed.append(payload['x'])
                    assert images_processed[-1].shape[0] == 1 or batches is not None
                    labels.append(payload['label'])
                    if len(images_processed) == batches:
                        break
                except tf.errors.OutOfRangeError:
                    break
    images_orig = np.concatenate(images_orig, axis=0)
    images_processed = np.concatenate(images_processed, axis=0)
    labels = np.concatenate(labels, axis=0)
    latents = [sess.run(ops.encode,
                        feed_dict={ops.x: images_processed[p:p + batch]})
               for p in range(0, images_processed.shape[0], FLAGS.batch)]
    latents = np.concatenate(latents, axis=0)
    latents = latents.reshape([latents.shape[0], -1])
    return images_orig, latents, labels


def save_split(split, X, Y, Z, ae):
    assert X.shape[0] == Y.shape[0] == Z.shape[0]
    assert X.dtype == np.uint8

    model = ae.__class__.__name__.lower()
    dataset = FLAGS.dataset[:-2]
    assert model in ['acai', 'aae', 'vqvae']
    assert dataset in ['mnist', 'celeba', 'miniimagenet', 'omniglot', 'miniimagenetgray']
    np.savez('./data/{}_{}_{}.npz'.format(dataset, Z.shape[-1], split), X=X, Y=Y, Z=Z)



def main(argv):
    del argv  # Unused.
    ae, ds = utils.load_ae(FLAGS.ae_dir, FLAGS.dataset, FLAGS.batch,
                           all_aes.ALL_AES, return_dataset=True)
    with utils.HookReport.disable():
        ae.eval_mode()

        # Convert all test samples to latents and get the labels
    val_images, val_latents, val_labels = get_images_and_latents_and_labels(ae.eval_sess,
                                                                            ae.eval_ops,
                                                                            ds.val)
    print('Shape of val_labels = {}'.format(np.shape(val_labels)))
    print('Shape of val_latents = {}'.format(np.shape(val_latents)))
    test_images, test_latents, test_labels = get_images_and_latents_and_labels(ae.eval_sess,
                                                                               ae.eval_ops,
                                                                               ds.test)
    print('Shape of test_labels = {}'.format(np.shape(test_labels)))
    print('Shape of test_latents = {}'.format(np.shape(test_latents)))
    train_images, train_latents, train_labels = get_images_and_latents_and_labels(ae.eval_sess,
                                                                                  ae.eval_ops,
                                                                                  ds.train_once)
    print('Shape of train_labels = {}'.format(np.shape(train_labels)))
    print('Shape of train_latents = {}'.format(np.shape(train_latents)))

    train_val_test_images = [train_images, val_images, test_images]
    train_val_test_labels = [train_labels, val_labels, test_labels]
    train_val_test_latents = [train_latents, val_latents, test_latents]

    # train_val_test_images = map(lambda x: unshift_and_unscale_all(x), train_val_test_images)
    # if FLAGS.dataset == 'mnist32' or FLAGS.dataset == 'omniglot32':
    #     train_val_test_images = map(lambda x: unpad(x), train_val_test_images)
    # if FLAGS.dataset == 'mnist32':
    #     train_val_test_labels = map(lambda y: np.argmax(y, axis=-1), train_val_test_labels)

    ipdb.set_trace()

    for (split, X, Y, Z) in zip(['train', 'val', 'test'], train_val_test_images, train_val_test_labels,
                                train_val_test_latents):
        save_split(split, X, Y, Z, ae)

if __name__ == '__main__':
    app.run(main)