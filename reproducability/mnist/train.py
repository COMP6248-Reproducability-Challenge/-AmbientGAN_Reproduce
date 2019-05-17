import tensorflow as tf
from config import *
from ambientGAN import *
from ops import *

Trainmode=True
def train(args, sess, model, RealDsIterator):
    # optimizers
    g_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer_G").minimize(
        model.g_loss, var_list=model.g_vars)
    d_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer_D").minimize(
        model.d_loss, var_list=model.d_vars)

    epoch = 0
    step = 0
    global_step = 0

    # saver
    saver = tf.train.Saver()
    if args.continue_training:
        last_ckpt = tf.train.latest_checkpoint(args.checkpoints_path)
        saver.restore(sess, last_ckpt)
        ckpt_name = str(last_ckpt)
        print("Loaded model file from " + ckpt_name)
        epoch = int(ckpt_name.split('-')[-1])
        tf.local_variables_initializer().run()
    else:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # summary init
    all_summary = tf.summary.merge([model.Y_r_sum,
                                    model.X_g_sum,
                                    model.Y_g_sum,
                                    model.d_loss_sum,
                                    model.g_loss_sum])
    writer = tf.summary.FileWriter(args.graph_path, sess.graph)

    # training starts here
    while epoch < args.epochs:
        batch_z = np.random.uniform(-1, 1, size=(args.batch_size, args.input_dim))

        x_real_o, y_real = RealDsIterator.next_batch(batch_size=args.batch_size)

        # model.X_o = model.measurement_fn(tf.convert_to_tensor(x_real))
        # x_real_ps = x_real_ps.numpy()
        # print(type(x_real_o), x_real_o.shape, x_real_o[0][0])
        # print(type(batch_z), batch_z.shape, batch_z[0])
        # Update Discriminator 1 times
        for _ in range(5):
            summary, d_loss, _ = sess.run([all_summary, model.d_loss, d_optimizer], \
                                          feed_dict={model.z: batch_z, model.X_o: x_real_o, model.label: y_real})
            writer.add_summary(summary, global_step)

        # Update Generator 2 times
        for _ in range(1):
            summary, g_loss, _ = sess.run([all_summary, model.g_loss, g_optimizer], feed_dict={model.z: batch_z, model.X_o: x_real_o, model.label: y_real})
            writer.add_summary(summary, global_step)

        print("Epoch [%d] Step [%d] G Loss: [%.4f] D Loss: [%.4f]" % (epoch, step, g_loss, d_loss))
        if step%100==0:
            saver.save(sess, args.checkpoints_path + "/model", global_step=epoch)

            res_img = sess.run(model.X_g, feed_dict={model.z: batch_z, model.X_o: x_real_o, model.label: y_real})
            res_img2 = sess.run(model.Y_r, feed_dict={model.z: batch_z, model.X_o: x_real_o, model.label: y_real})

            img_tile(epoch, step, args, res_img)
            img_tile_2(epoch, step, args, res_img2)

        if step * args.batch_size >= model.data_count:
            saver.save(sess, args.checkpoints_path + "/model", global_step=epoch)

            res_img = sess.run(model.X_g, feed_dict={model.z: batch_z, model.X_o: x_real_o, model.label: y_real})

            img_tile(epoch,step, args, res_img)
            step = 0
            epoch += 1

        step += 1
        global_step += 1
        #total steps
        if global_step>27000:
            break

    coord.request_stop()
    coord.join(threads)
    sess.close()
    print("Done.")


def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        model = ambientGAN(args,Trainmode)
        args.images_path = os.path.join(args.images_path, args.measurement)
        args.graph_path = os.path.join(args.graph_path, args.measurement)
        args.checkpoints_path = os.path.join(args.checkpoints_path, args.measurement)

        # create graph, images, and checkpoints folder if they don't exist
        if not os.path.exists(args.checkpoints_path):
            os.makedirs(args.checkpoints_path)
        if not os.path.exists(args.graph_path):
            os.makedirs(args.graph_path)
        if not os.path.exists(args.images_path):
            os.makedirs(args.images_path)

        real_dataset_iterator = RealDsIterator()

        print('Start Training...')
        train(args, sess, model, real_dataset_iterator)


main(args)

# Still Working....