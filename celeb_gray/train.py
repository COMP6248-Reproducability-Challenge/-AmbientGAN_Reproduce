import tensorflow as tf
from config import *
from ambientGAN import *
import time

def train(args, sess, model):
    #optimizers
    g_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer_G").minimize(model.g_loss, var_list=model.g_vars)
    d_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer_D").minimize(model.d_loss, var_list=model.d_vars)
    init_start = time.time()


    epoch = 0
    step = 0
    global_step = 0

    #saver
    saver = tf.train.Saver()        
    if args.continue_training:
        last_ckpt = tf.train.latest_checkpoint(args.checkpoints_path)
        saver.restore(sess, last_ckpt)
        ckpt_name = str(last_ckpt)
        print "Loaded model file from " + ckpt_name
        epoch = int(ckpt_name.split('-')[-1])
        tf.local_variables_initializer().run()
    else:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #summary init
    all_summary = tf.summary.merge([model.Y_r_sum,
                                    model.X_g_sum,
                                    model.Y_g_sum, 
                                    model.d_loss_sum,
                                    model.g_loss_sum])
    writer = tf.summary.FileWriter(args.graph_path, sess.graph)

    #training starts here
    while epoch < args.epochs:
        start_time = time.time()
        batch_z = np.random.uniform(-1, 1, size=(args.batch_size , args.input_dim))

        #Update Discriminator
        summary, d_loss, _ = sess.run([all_summary, model.d_loss, d_optimizer], feed_dict={model.z:batch_z})
        writer.add_summary(summary, global_step)

        #Update Generator
        summary, g_loss, _ = sess.run([all_summary, model.g_loss, g_optimizer], feed_dict={model.z:batch_z})
        writer.add_summary(summary, global_step)
        #Update Generator Again
        summary, g_loss, _ = sess.run([all_summary, model.g_loss, g_optimizer], feed_dict={model.z:batch_z})
        writer.add_summary(summary, global_step)


        print "Epoch [%d] Step [%d] G Loss: [%.4f] D Loss: [%.4f]--- [%s] seconds ---" % (epoch, step, g_loss, d_loss,time.time() - start_time)

        if step*args.batch_size >= model.data_count:
            saver.save(sess, args.checkpoints_path + "/model", global_step=epoch)

            res_img = sess.run(model.X_g, feed_dict={model.z:batch_z})
            train_img = sess.run(model.Y_r)
            img_tile(epoch, args, res_img)
            img_tile_lossy_input(epoch,args,train_img)
            step = 0
            epoch += 1
            print "total [%s]" % (time.time() - init_start)

        step += 1
        global_step += 1



    coord.request_stop()
    coord.join(threads)
    sess.close()            
    print("Done.")


def main(_):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    init = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()

    with tf.Session(config=run_config) as sess:
        sess.run(init)
        sess.run(init_l)
        model = ambientGAN(args)
        args.images_path = os.path.join(args.images_path, args.measurement)
        args.graph_path = os.path.join(args.graph_path, args.measurement)
        args.checkpoints_path = os.path.join(args.checkpoints_path, args.measurement)

        #create graph, images, and checkpoints folder if they don't exist
        if not os.path.exists(args.checkpoints_path):
            os.makedirs(args.checkpoints_path)
        if not os.path.exists(args.graph_path):
            os.makedirs(args.graph_path)
        if not os.path.exists(args.images_path):
            os.makedirs(args.images_path)

        print 'Start Training...'
        train(args, sess, model)

main(args)

#Still Working....