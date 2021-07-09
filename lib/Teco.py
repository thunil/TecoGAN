from lib.frvsr import *

VGG_MEAN = [123.68, 116.78, 103.94]

def VGG19_slim(input, reuse, deep_list=None, norm_flag=True):
    # deep_list, define the feature to extract
    with tf.device('/gpu:0'):
        # preprocessing:
        input_img = deprocess(input)
        input_img_ab = input_img * 255.0 - tf.constant(VGG_MEAN)
        # model:
        _, output = vgg_19(input_img_ab, is_training=False, reuse=reuse)
        # feature maps:
        results = {}
        with tf.name_scope('vgg_norm'):
            for key in output:
                if (deep_list is None) or (key in deep_list):
                    orig_deep_feature = output[key]
                    if norm_flag:
                        orig_len = tf.sqrt(tf.reduce_sum(tf.square(orig_deep_feature), axis=[3], keepdims=True)+1e-12)
                        results[key] = orig_deep_feature / orig_len
                    else:
                        results[key] = orig_deep_feature
    return results
    
    
# Definition of the discriminator,
# the structure here is a normal structure, 
# its inputs extend it to a spatio-temporal D, when using in TecoGAN function
def discriminator_F(dis_inputs, FLAGS=None):
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for generator')

    # Define the discriminator block
    def discriminator_block(inputs, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(inputs, kernel_size, output_channel, stride, use_bias=False, scope='conv1')
            net = batchnorm(net, is_training=True)
            net = lrelu(net, 0.2)

        return net
        
    layer_list = []
    with tf.device('/gpu:0'), tf.variable_scope('discriminator_unit'):
        # The input layer
        with tf.variable_scope('input_stage'):
            # no batchnorm for the first layer
            net = conv2(dis_inputs, 3, 64, 1, scope='conv')
            net = lrelu(net, 0.2) # (b, h,w,64)
            

        # The discriminator block part
        # block 1
        net = discriminator_block(net, 64, 4, 2, 'disblock_1')
        layer_list += [net] # (b, h/2,w/2,64)

        # block 2
        net = discriminator_block(net, 64, 4, 2, 'disblock_3')
        layer_list += [net] # (b, h/4,w/4,64)

        # block 3
        net = discriminator_block(net, 128, 4, 2, 'disblock_5')
        layer_list += [net] # (b, h/8,w/8,128)

        # block_4
        net = discriminator_block(net, 256, 4, 2, 'disblock_7')
        layer_list += [net]  # (b, h/16,w/16,256)

        # The dense layer 1
        with tf.variable_scope('dense_layer_2'):
            net = denselayer(net, 1) # channel-wise dense layer
            net = tf.nn.sigmoid(net) # (h/16,w/16,1)

    return net, layer_list
    
# Define the whole network architecture
def TecoGAN(r_inputs, r_targets, FLAGS, GAN_Flag=True):
    # r_inputs, r_targets : shape (b,frame,h,w,c)
    inputimages = FLAGS.RNN_N
    if FLAGS.pingpang:# Ping-Pang reuse
        r_inputs_rev_input = r_inputs[:,-2::-1,:,:,:]
        r_targets_rev_input = r_targets[:,-2::-1,:,:,:]
        r_inputs = tf.concat([r_inputs, r_inputs_rev_input],axis = 1)
        r_targets = tf.concat([r_targets, r_targets_rev_input],axis = 1)
        inputimages = FLAGS.RNN_N * 2 - 1
    
    # output_channel, the last channel number
    output_channel = r_targets.get_shape().as_list()[-1]
    
    # list for all outputs, and warped previous frames
    gen_outputs, gen_warppre = [], []
    # gen_warppre is not necessary, just for showing in tensorboard
    
    # Define the learning rate and global step
    with tf.variable_scope('get_learning_rate_and_global_step'):
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, 
                        global_step, FLAGS.decay_step, FLAGS.decay_rate, staircase=FLAGS.stair)
        incr_global_step = tf.assign(global_step, global_step + 1)
        
    # Build the generator part, fnet
    with tf.device('/gpu:0'), tf.variable_scope('fnet'):
        Frame_t_pre = r_inputs[:, 0:-1, :,:,:] 
        # batch, frame-1, FLAGS.crop_size, FLAGS.crop_size, output_channel
        Frame_t = r_inputs[:, 1:, :,:,:] 
        # batch, frame-1, FLAGS.crop_size, FLAGS.crop_size, output_channel
        
        fnet_input = tf.concat( (Frame_t_pre, Frame_t), axis = -1 )
        fnet_input = tf.reshape( fnet_input, (FLAGS.batch_size*(inputimages-1), FLAGS.crop_size, FLAGS.crop_size, 2*output_channel) )
        # batch*(frame-1), FLAGS.crop_size, FLAGS.crop_size, output_channel
        gen_flow_lr = fnet( fnet_input, reuse=False ) 
        # batch * (inputimages-1), FLAGS.crop_size, FLAGS.crop_size, 2
        gen_flow = upscale_four(gen_flow_lr*4.0) # a linear up-sampling
        # batch * (inputimages-1), FLAGS.crop_size*4, FLAGS.crop_size*4, 2
        gen_flow = tf.reshape( gen_flow, (FLAGS.batch_size,(inputimages-1), FLAGS.crop_size*4, FLAGS.crop_size*4, 2) )
        # Compute the euclidean distance between the two features (input_frames and s_input_warp) as the warp_loss
        input_frames = tf.reshape( Frame_t, (FLAGS.batch_size*(inputimages-1), FLAGS.crop_size, FLAGS.crop_size, output_channel) )
        
    # tf.contrib.image.dense_image_warp, only in tf1.8 or larger, no GPU support
    s_input_warp = tf.contrib.image.dense_image_warp( 
        tf.reshape( Frame_t_pre, (FLAGS.batch_size*(inputimages-1), FLAGS.crop_size, FLAGS.crop_size, output_channel) ),
        gen_flow_lr) # (FLAGS.batch_size*(inputimages-1), FLAGS.crop_size, FLAGS.crop_size, output_channel)
    
    # Build the generator part, a recurrent generator
    with tf.variable_scope('generator'):
        # for the first frame, concat with zeros
        input0 = tf.concat( \
            ( r_inputs[:,0,:,:,:], tf.zeros((FLAGS.batch_size, FLAGS.crop_size, FLAGS.crop_size, 3*4*4), \
            dtype=tf.float32) ), axis = -1)
        gen_pre_output = generator_F(input0, output_channel, reuse=False, FLAGS=FLAGS)
        # batch, FLAGS.crop_size*4, FLAGS.crop_size*4, 3
        
        gen_pre_output.set_shape( (FLAGS.batch_size, FLAGS.crop_size*4, FLAGS.crop_size*4, 3) )
        gen_outputs.append(gen_pre_output) # frame 0, done
        
        for frame_i in range( inputimages - 1 ):
            # warp the previously generated frame
            cur_flow = gen_flow[:, frame_i, :,:,:]
            cur_flow.set_shape( (FLAGS.batch_size, FLAGS.crop_size*4, FLAGS.crop_size*4, 2) )
            gen_pre_output_warp = tf.contrib.image.dense_image_warp(
                gen_pre_output, cur_flow)
            gen_warppre.append(gen_pre_output_warp) # warp frame [0,n-1] to frame [1,n]
            gen_pre_output_warp = preprocessLR( deprocess(gen_pre_output_warp) )
            # apply space-to-depth transform
            gen_pre_output_reshape = tf.reshape(gen_pre_output_warp, (FLAGS.batch_size, FLAGS.crop_size, 4, FLAGS.crop_size, 4, 3) )
            gen_pre_output_reshape = tf.transpose( gen_pre_output_reshape, perm = [0,1,3,2,4,5] )
            # batch,FLAGS.crop_size, FLAGS.crop_size, 4, 4, 3
            gen_pre_output_reshape = tf.reshape(gen_pre_output_reshape, (FLAGS.batch_size, FLAGS.crop_size, FLAGS.crop_size, 3*4*4) )
            # pack it as the recurrent input
            inputs = tf.concat( ( r_inputs[:,frame_i+1,:,:,:], gen_pre_output_reshape ), axis = -1)
            # super-resolution part
            gen_output = generator_F(inputs, output_channel, reuse=True, FLAGS=FLAGS)
            gen_outputs.append(gen_output)
            gen_pre_output = gen_output
            gen_pre_output.set_shape( (FLAGS.batch_size, FLAGS.crop_size*4, FLAGS.crop_size*4, 3) )
            
        # gen_outputs, a list, len = frame, shape = (batch, FLAGS.crop_size*4, FLAGS.crop_size*4, 3)
        gen_outputs = tf.stack( gen_outputs, axis = 1 ) 
        # batch, frame, FLAGS.crop_size*4, FLAGS.crop_size*4, 3
        gen_outputs.set_shape([FLAGS.batch_size, inputimages, FLAGS.crop_size*4, FLAGS.crop_size*4, 3])
        # gen_warppre holds all the warped previous frames
        gen_warppre = tf.stack( gen_warppre, axis = 1 ) 
        # batch, frame-1, FLAGS.crop_size*4, FLAGS.crop_size*4, 3
        gen_warppre.set_shape([FLAGS.batch_size, inputimages-1, FLAGS.crop_size*4, FLAGS.crop_size*4, 3])
            
        
    # these are necessary for losses
    s_gen_output = tf.reshape(gen_outputs, (FLAGS.batch_size*inputimages, FLAGS.crop_size*4, FLAGS.crop_size*4, 3) )
    s_targets = tf.reshape(r_targets, (FLAGS.batch_size*inputimages, FLAGS.crop_size*4, FLAGS.crop_size*4, 3) )
            
    update_list = [] # collect all the losses, show the original value before scaling
    update_list_name = [] # collect all the names of the losses
    
    if FLAGS.vgg_scaling > 0.0:
        with tf.name_scope('vgg_19') as scope:
            vgg_layer_labels = ['vgg_19/conv2/conv2_2', 'vgg_19/conv3/conv3_4', 'vgg_19/conv4/conv4_4', 'vgg_19/conv5/conv5_4'] 
            gen_vgg = VGG19_slim(s_gen_output, reuse=False, deep_list=vgg_layer_labels)
            target_vgg = VGG19_slim(s_targets, reuse=True, deep_list=vgg_layer_labels)
            
    if(GAN_Flag):# build the discriminator
        # prepare inputs
        with tf.device('/gpu:0'), tf.name_scope('input_Tdiscriminator'):
            t_size = int( 3 * (inputimages // 3) ) # 3 frames are used as one entry, the last inputimages%3 frames are abandoned
            t_gen_output = tf.reshape(gen_outputs[:,:t_size,:,:,:], 
                (FLAGS.batch_size*t_size, FLAGS.crop_size*4, FLAGS.crop_size*4, 3) )
            t_targets = tf.reshape(r_targets[:,:t_size,:,:,:], 
                (FLAGS.batch_size*t_size, FLAGS.crop_size*4, FLAGS.crop_size*4, 3) )
            t_batch = FLAGS.batch_size*t_size//3
            
            if not FLAGS.pingpang: # backward motion has to be calculated
                fnet_input_back = tf.concat( (r_inputs[:, 2:t_size:3, :,:,:], r_inputs[:, 1:t_size:3, :,:,:] ), axis = -1 )
                fnet_input_back = tf.reshape( fnet_input_back, (t_batch, FLAGS.crop_size, FLAGS.crop_size, 2*output_channel) )
                
                with tf.variable_scope('fnet'):
                    gen_flow_back_lr = fnet( fnet_input_back, reuse=True ) 
                    # t_batch, FLAGS.crop_size, FLAGS.crop_size, 2
                    gen_flow_back = upscale_four(gen_flow_back_lr*4.0)
                # t_batch, FLAGS.crop_size*4, FLAGS.crop_size*4, 2
                gen_flow_back = tf.reshape( gen_flow_back, (FLAGS.batch_size, t_size//3, FLAGS.crop_size*4, FLAGS.crop_size*4, 2) )
                
                T_inputs_VPre_batch = tf.identity(gen_flow[:,0:t_size:3,:,:,:]) # forward motion reused, 
                T_inputs_V_batch = tf.zeros_like(T_inputs_VPre_batch) # no motion for middle frames, todo remove for better performance
                T_inputs_VNxt_batch = tf.identity(gen_flow_back) # backward motion
                # The above are all in shape of (FLAGS.batch_size, t_size//3, crop_size, crop_size, 2)
                
            else:# motion could be reused for Ping-pang sequence, also all in shape of (FLAGS.batch_size, t_size//3, crop_size, crop_size, 2)
                T_inputs_VPre_batch = tf.identity(gen_flow[:,0:t_size:3,:,:,:]) # forward motion reused, 
                T_inputs_V_batch = tf.zeros_like(T_inputs_VPre_batch) # no motion for middle frames, 
                T_inputs_VNxt_batch = tf.identity(gen_flow[:,-2:-1-t_size:-3,:,:,:]) # backward motion reused
            
            T_vel = tf.stack( [T_inputs_VPre_batch, T_inputs_V_batch, T_inputs_VNxt_batch], axis = 2 )
            # batch, t_size/3, 3, FLAGS.crop_size*4, FLAGS.crop_size*4, 2
            T_vel = tf.reshape(T_vel, (FLAGS.batch_size*t_size, FLAGS.crop_size*4, FLAGS.crop_size*4, 2) )
            T_vel = tf.stop_gradient( T_vel ) # won't passing gradient to fnet from discriminator, details in TecoGAN supplemental paper 
            
        if(FLAGS.crop_dt < 1.0): # crop out unstable part for temporal discriminator, details in TecoGAN supplemental paper 
            crop_size_dt = int( FLAGS.crop_size * 4 * FLAGS.crop_dt)
            offset_dt = (FLAGS.crop_size * 4 - crop_size_dt) // 2
            crop_size_dt = FLAGS.crop_size * 4 - offset_dt*2
            paddings =  tf.constant([[0, 0], [offset_dt, offset_dt], [offset_dt, offset_dt],[0,0]])
        
        # Build the tempo discriminator for the real part
        with tf.name_scope('real_Tdiscriminator'):
            real_warp0 = tf.contrib.image.dense_image_warp(t_targets, T_vel) 
            # batch*t_size, h=FLAGS.crop_size*4, w=FLAGS.crop_size*4, 3
            with tf.device('/gpu:0'), tf.variable_scope('tdiscriminator', reuse=False):
                real_warp = tf.reshape(real_warp0, (t_batch, 3, FLAGS.crop_size*4, FLAGS.crop_size*4, 3))#[tb,T=3,h,w,ch=3 for RGB]
                real_warp = tf.transpose(real_warp, perm=[0, 2, 3, 4, 1]) # [tb,h,w,RGB,3T], 3T are t-1, t and t+1
                real_warp = tf.reshape(real_warp, (t_batch,FLAGS.crop_size*4, FLAGS.crop_size*4,9) )# [tb,h,w,RRRGGGBBB] RRR: Red_t-1, Red_t, Red_t+1
                if(FLAGS.crop_dt < 1.0):
                    real_warp = tf.image.crop_to_bounding_box(real_warp, offset_dt, offset_dt, crop_size_dt, crop_size_dt)
                    
                if(FLAGS.Dt_mergeDs): # a spatio-temporal D
                    if(FLAGS.crop_dt < 1.0): real_warp = tf.pad(real_warp, paddings, "CONSTANT" )
                    with tf.variable_scope('sdiscriminator', reuse=False): # actually no more variable under this scope... 
                        before_warp = tf.reshape(t_targets, (t_batch, 3, FLAGS.crop_size*4, FLAGS.crop_size*4, 3))#[tb,3=T,h,w,3=RGB]
                        before_warp = tf.transpose(before_warp, perm=[0, 2, 3, 4, 1]) # [b,h,w,3RGB,3T]
                        before_warp = tf.reshape(before_warp, (t_batch, FLAGS.crop_size*4, FLAGS.crop_size*4, 3*3)) # [b,h,w,9=RRRGGGBBB]
                        
                        t_input = tf.reshape(r_inputs[:,:t_size,:,:,:], \
                            (t_batch, 3, FLAGS.crop_size, FLAGS.crop_size, -1) )
                        t_input = tf.transpose(t_input, perm=[0, 2, 3, 4, 1]) # [tb,h//4,w//4,3RGB,3T]
                        t_input = tf.reshape(t_input, (t_batch, FLAGS.crop_size, FLAGS.crop_size,-1) ) # [tb,h//4,w//4,9=RRRGGGBBB]
                        input_hi = tf.image.resize_images( t_input, (FLAGS.crop_size*4, FLAGS.crop_size*4) )# [tb,h,w,9=RRRGGGBBB]
                        real_warp = tf.concat( (before_warp, real_warp, input_hi), axis = -1)# [tb,h,w,9 + 9 + 9]
                        
                    tdiscrim_real_output, real_layers = discriminator_F(real_warp, FLAGS=FLAGS)
                    
                else: # an unconditional Dt
                    tdiscrim_real_output = discriminator_F(real_warp, FLAGS=FLAGS)# [tb,h*FLAGS.crop_dt,w*FLAGS.crop_dt,RRRGGGBBB]
                    
        # Build the tempo discriminator for the fake part
        with tf.name_scope('fake_Tdiscriminator'):
            fake_warp0 = tf.contrib.image.dense_image_warp(t_gen_output, T_vel)
            with tf.device('/gpu:0'), tf.variable_scope('tdiscriminator', reuse=True): # reuse weights
                fake_warp = tf.reshape(fake_warp0, (t_batch, 3, FLAGS.crop_size*4, FLAGS.crop_size*4, 3))
                fake_warp = tf.transpose(fake_warp, perm=[0, 2, 3, 4, 1])
                fake_warp = tf.reshape(fake_warp, (t_batch,FLAGS.crop_size*4, FLAGS.crop_size*4,9) )
                if(FLAGS.crop_dt < 1.0):
                    fake_warp = tf.image.crop_to_bounding_box(fake_warp, offset_dt, offset_dt, crop_size_dt, crop_size_dt)
                    
                if(FLAGS.Dt_mergeDs): # a spatio-temporal D
                    if(FLAGS.crop_dt < 1.0): fake_warp = tf.pad(fake_warp, paddings, "CONSTANT" )
                    with tf.variable_scope('sdiscriminator', reuse=True): # actually no more variable under this scope... 
                        before_warp = tf.reshape(t_gen_output, (t_batch, 3, FLAGS.crop_size*4, FLAGS.crop_size*4, 3))
                        before_warp = tf.transpose(before_warp, perm=[0, 2, 3, 4, 1])
                        before_warp = tf.reshape(before_warp, (t_batch, FLAGS.crop_size*4, FLAGS.crop_size*4, 3*3))
                        # input_hi is reused
                        fake_warp = tf.concat( (before_warp, fake_warp, input_hi), axis = -1) # [tb,h,w,9 + 9 + 9]
                    tdiscrim_fake_output, fake_layers = discriminator_F(fake_warp, FLAGS=FLAGS)
                else:
                    tdiscrim_fake_output = discriminator_F(fake_warp, FLAGS=FLAGS)
                    
        # prepare the layer between discriminators
        if(FLAGS.D_LAYERLOSS): 
            # The parameters here are hard coded here, just to roughly scale the 
            #   layer losses to a similar level, around 'Fix_Range',
            #   so that every layer is playing a role.
            # A better fine-tuning could improve things.
            with tf.device('/gpu:0'), tf.variable_scope('layer_loss'):
                Fix_Range = 0.02 # hard coded, all layers are roughly scaled to this value
                Fix_margin = 0.0 # 0.0 will ignore losses on the Discriminator part, which is good,
                                 # because it is too strong usually. details in paper 
                sum_layer_loss = 0 # adds-on for generator
                d_layer_loss = 0 # adds-on for discriminator, clipped with Fix_margin
                
                layer_loss_list = []
                layer_n = len(real_layers)
                
                layer_norm = [12.0, 14.0, 24.0, 100.0] # hard coded, an overall average of all layers
                for layer_i in range(layer_n):
                    real_layer = real_layers[layer_i]
                    false_layer = fake_layers[layer_i]
                    
                    layer_diff = real_layer - false_layer
                    layer_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(layer_diff), axis=[3]))
                    # an l1 loss
                    layer_loss_list += [layer_loss]
                    
                    scaled_layer_loss = Fix_Range * layer_loss / layer_norm[layer_i]
                    
                    sum_layer_loss += scaled_layer_loss
                    if Fix_margin > 0.0:
                        d_layer_loss += tf.maximum( 0.0, Fix_margin - scaled_layer_loss)
                    
                update_list += layer_loss_list
                update_list_name += [("D_layer_%d_loss" % _) for _ in range(layer_n)]
                update_list += [sum_layer_loss]
                update_list_name += ["D_layer_loss_sum"] # for G
                
                if Fix_margin > 0.0:
                    update_list += [d_layer_loss]
                    update_list_name += ["D_layer_loss_for_D_sum"]

    # Build the loss
    with tf.variable_scope('generator_loss'):
        # Content loss, l2 loss 
        with tf.device('/gpu:0'), tf.variable_scope('content_loss'):
            # Compute the euclidean distance between the two features
            diff1_mse = s_gen_output - s_targets
            # (FLAGS.batch_size*(inputimages), FLAGS.crop_sizex4, FLAGS.crop_sizex4, 3)
            content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff1_mse), axis=[3]))
            update_list += [content_loss] # an l2 loss
            update_list_name += ["l2_content_loss"]
            gen_loss = content_loss
        
        # Warp loss
        with tf.variable_scope('warp_loss'):
            diff2_mse = input_frames - s_input_warp 
            # (FLAGS.batch_size*(inputimages), FLAGS.crop_size, FLAGS.crop_size, 3)
            warp_loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff2_mse), axis=[3]))
            update_list += [warp_loss]
            update_list_name += ["l2_warp_loss"]
            # the following line is useless, because low-res warping has no gradient on generator
            # gen_loss += FLAGS.warp_scaling * warp_loss 
                        
        vgg_loss = None
        vgg_loss_list = []
        if FLAGS.vgg_scaling > 0.0:
            with tf.device('/gpu:0'), tf.variable_scope('vgg_layer_loss'):
                # we use 4 VGG layers
                vgg_wei_list = [1.0,1.0,1.0,1.0] 
                vgg_loss = 0
                vgg_layer_n = len(vgg_layer_labels)
                
                for layer_i in range(vgg_layer_n):
                    curvgg_diff = tf.reduce_sum(gen_vgg[vgg_layer_labels[layer_i]]*target_vgg[vgg_layer_labels[layer_i]], axis=[3])
                    # cosine similarity, -1~1, 1 best
                    curvgg_diff = 1.0 - tf.reduce_mean(curvgg_diff) # 0 ~ 2, 0 best
                    scaled_layer_loss = vgg_wei_list[layer_i] * curvgg_diff
                    vgg_loss_list += [curvgg_diff]
                    vgg_loss += scaled_layer_loss
                    
            gen_loss += FLAGS.vgg_scaling * vgg_loss
            vgg_loss_list += [vgg_loss]
            
            update_list += vgg_loss_list
            update_list_name += ["vgg_loss_%d"%(_+2) for _ in range(len(vgg_loss_list)-1)]
            update_list_name += ["vgg_all"]
            
        # Here is the Ping-pang loss
        if FLAGS.pingpang:
            with tf.device('/gpu:0'), tf.variable_scope('bidirection_loss'):
                gen_out_first = gen_outputs[:,0:FLAGS.RNN_N-1,:,:,:]
                gen_out_last_rev = gen_outputs[:,-1:-FLAGS.RNN_N:-1,:,:,:]
                # an l1 loss
                pploss = tf.reduce_mean(tf.abs(gen_out_first - gen_out_last_rev))
            
            if FLAGS.pp_scaling > 0:
                gen_loss += pploss * FLAGS.pp_scaling
            update_list += [pploss]
            update_list_name += ["PingPang"]
        
        if(GAN_Flag): # spatio-temporal adversarial loss
            with tf.variable_scope('t_adversarial_loss'):
                t_adversarial_loss = tf.reduce_mean(-tf.log(tdiscrim_fake_output + FLAGS.EPS))
                # we can fade in of the discrim_loss, 
                # but for TecoGAN paper, we always use FLAGS.Dt_ratio_0 and Dt_ratio_max as 1.0 (no fading in)
                dt_ratio = tf.minimum( FLAGS.Dt_ratio_max, \
                    FLAGS.Dt_ratio_0 + FLAGS.Dt_ratio_add * tf.cast(global_step, tf.float32) )
                        
                t_adversarial_fading = t_adversarial_loss * dt_ratio
            
            gen_loss += FLAGS.ratio * t_adversarial_fading
            update_list += [t_adversarial_loss]
            update_list_name += ["t_adversarial_loss"]
            
            # layer loss from discriminator
            if(FLAGS.D_LAYERLOSS):
                gen_loss += sum_layer_loss * dt_ratio # positive layer loss, with fading in as well
        
    if(GAN_Flag): # Build the discriminator loss
        with tf.device('/gpu:0'), tf.variable_scope('t_discriminator_loss'):
            t_discrim_fake_loss = tf.log(1 - tdiscrim_fake_output + FLAGS.EPS)
            t_discrim_real_loss = tf.log(tdiscrim_real_output + FLAGS.EPS)
            
            t_discrim_loss = tf.reduce_mean(-(t_discrim_fake_loss + t_discrim_real_loss))\
            # a criterion of updating Dst
            t_balance = tf.reduce_mean(t_discrim_real_loss) + t_adversarial_loss
            # if t_balance is very large (>0.4), it means the discriminator is too strong
            
            update_list += [t_discrim_loss]
            update_list_name += ["t_discrim_loss"]
            
            update_list += [tf.reduce_mean(tdiscrim_real_output), tf.reduce_mean(tdiscrim_fake_output)]
            update_list_name += ["t_discrim_real_output", "t_discrim_fake_output"]
            
            if(FLAGS.D_LAYERLOSS and Fix_margin>0.0 ):
                discrim_loss = t_discrim_loss + d_layer_loss * dt_ratio  
                # hinge negative layer loss, with fading in as well
            else:
                discrim_loss = t_discrim_loss
        
        # use a second exp_averager, because updating time is different to loss summary
        tb_exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
        update_tb = tb_exp_averager.apply([t_balance]) 
        tb = tb_exp_averager.average(t_balance)
        
        # Build the discriminator train
        with tf.device('/gpu:0'), tf.variable_scope('tdicriminator_train'):
            tdiscrim_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='tdiscriminator')
            tdis_learning_rate = learning_rate
            if( not FLAGS.Dt_mergeDs ):# use a smaller learning rate when Dt only (Hard-coded as 0.3), otherwise blur too much
                tdis_learning_rate = tdis_learning_rate * 0.3
            tdiscrim_optimizer = tf.train.AdamOptimizer(tdis_learning_rate, beta1=FLAGS.beta, epsilon=FLAGS.adameps)
            tdiscrim_grads_and_vars = tdiscrim_optimizer.compute_gradients(discrim_loss, tdiscrim_tvars)
            # https://github.com/tensorflow/tensorflow/issues/3287
            # tdiscrim_train = tf.cond(tf.less(tb, 0.4), lambda: tf.group(tdiscrim_train, update_tb), lambda: update_tb)
    
    update_list += [gen_loss]
    update_list_name += ["All_loss_Gen"]
    # a moving average to collect all training statistics
    exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
    update_loss = exp_averager.apply(update_list)
    update_list_avg = [exp_averager.average(_) for _ in update_list]
        
    # Build the Adam_train and Return the network
    with tf.variable_scope('generator_train'):
        gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta, epsilon=FLAGS.adameps)
        fnet_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta, epsilon=FLAGS.adameps)
        gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        fnet_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fnet')
        fnet_loss = FLAGS.warp_scaling * warp_loss + gen_loss
            
        if(not GAN_Flag):
            gen_grads_and_vars = gen_optimizer.compute_gradients(gen_loss, gen_tvars)
            fnet_grads_and_vars = fnet_optimizer.compute_gradients(fnet_loss, fnet_tvars)
            gen_train = gen_optimizer.apply_gradients(gen_grads_and_vars)
            fnet_train = fnet_optimizer.apply_gradients(fnet_grads_and_vars)
        else:
            update_list_avg += [tb, dt_ratio]
            update_list_name += ["t_balance", "Dst_ratio"]
            # Need to wait discriminator to perform train step
            # tf.GraphKeys.UPDATE_OPS: batch normalization layer in discriminator should update first
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                counter1 = tf.get_variable( dtype=tf.int32, shape=(), name='gen_train_with_D_counter',\
                    initializer=tf.zeros_initializer())
                counter2 = tf.get_variable( dtype=tf.int32, shape=(), name='gen_train_wo_D_counter',\
                    initializer=tf.zeros_initializer())
                
                def train_gen_withD():
                    ''' train generator with discriminator '''                    
                    tdiscrim_train = tdiscrim_optimizer.apply_gradients(tdiscrim_grads_and_vars)                    
                    with tf.control_dependencies([ tf.assign_add(counter1, 1), update_tb, tdiscrim_train ]):
                        gen_grads_and_vars1 = gen_optimizer.compute_gradients(gen_loss, gen_tvars)
                        fnet_grads_and_vars1 = fnet_optimizer.compute_gradients(fnet_loss, fnet_tvars)                    
                        gen_train1 = gen_optimizer.apply_gradients(gen_grads_and_vars1)
                        fnet_train1 = fnet_optimizer.apply_gradients(fnet_grads_and_vars1)
                    return gen_train1, fnet_train1, gen_grads_and_vars1, fnet_grads_and_vars1
                    
                    
                def train_gen_withoutD():
                    ''' 
                    train generator without discriminator, 
                    sometimes important, when discriminator is too good
                    '''
                    with tf.control_dependencies([ tf.assign_add(counter2, 1), update_tb ]):
                        gen_grads_and_vars2 = gen_optimizer.compute_gradients(gen_loss, gen_tvars)
                        fnet_grads_and_vars2 = fnet_optimizer.compute_gradients(fnet_loss, fnet_tvars)                    
                        gen_train2 = gen_optimizer.apply_gradients(gen_grads_and_vars2)
                        fnet_train2 = fnet_optimizer.apply_gradients(fnet_grads_and_vars2)

                    return gen_train2, fnet_train2, gen_grads_and_vars2, fnet_grads_and_vars2
                
                # train D once, G twice
                # gen_train, fnet_train, gen_grads_and_vars, fnet_grads_and_vars = \
                #    tf.cond( tf.equal(tf.floormod(global_step, 2), tf.constant(0,tf.int64)), train_gen_withD, train_gen_withoutD )

                # train D once, G once, 
                # gen_train, fnet_train, gen_grads_and_vars, fnet_grads_and_vars = train_gen_withD()
                
                # train D adaptively
                gen_train, fnet_train, gen_grads_and_vars, fnet_grads_and_vars = \
                     tf.cond( tf.less(tb, FLAGS.Dbalance), train_gen_withD, train_gen_withoutD )
                update_list_avg += [counter1, counter2]
                update_list_name += ["withD_counter", "w_o_D_counter"]
            
    with tf.name_scope('image_summaries'):
        max_outputs = min(4, FLAGS.batch_size)
        gif_sum = [ gif_summary('LR', r_inputs, max_outputs=max_outputs, fps=3),
                gif_summary('HR', deprocess(r_targets), max_outputs=max_outputs, fps=3),
                gif_summary('Generated', deprocess(gen_outputs), max_outputs=max_outputs, fps=3),
                gif_summary('WarpPreGen', deprocess(gen_warppre), max_outputs=max_outputs, fps=3),]
        # todo add fake_warp and real_warp in gif_sum as well
        
    Network = collections.namedtuple('Network', 'gen_output, train, learning_rate, update_list, '
                                         'update_list_name, update_list_avg, image_summary, global_step')
    return Network(
        gen_output = s_gen_output,
        train = tf.group(update_loss, incr_global_step, gen_train, fnet_train),
        learning_rate = learning_rate,
        update_list = update_list,
        update_list_name = update_list_name,
        update_list_avg = update_list_avg,
        image_summary = gif_sum, 
        global_step = global_step,
    )
        
    
# FRVSR alias
def FRVSR(r_inputs, r_targets, FLAGS):
    return TecoGAN(r_inputs, r_targets, FLAGS, False)