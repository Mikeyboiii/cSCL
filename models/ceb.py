def add_contrastive_ceb_loss(hidden,
                             kappa_e=1024.0,
                             kappa_b=10.0,
                             beta=1.0,
                             bidirectional=True,
                             sampling=True,
                             double_batch_trick=True,
                             strategy=None):
  """Compute contrastive version of CEB loss (von Mises-Fisher) for model.
  Args:
    hidden: hidden vector (`Tensor`) of shape (B, dim).
    kappa_e: forward encoder concentration.
    kappa_b: backward encoder concentration.
    beta: CEB beta.
    bidirectional: whether to compute loss in a bidirectional manner.
    sampling: whether sampling from forward encoder or not.
    double_batch_trick: whether to use negatives from encoder e to
      double the batch size.
    strategy: a `tf.distribute.Strategy`.
  Returns:
    A loss scalar.
    The logits for contrastive prediction task.
    The labels for contrastive prediction task.
    A metrics dictionary.
  """
  # Get (normalized) hidden1 and hidden2.
  hidden = tf.math.l2_normalize(hidden, -1)
  hidden1, hidden2 = tf.split(hidden, 2, 0)
  batch_size = tf.shape(hidden1)[0]

  # Gather hidden1/hidden2 across replicas and create local labels.
  if strategy is not None:
    hidden1_large = tpu_cross_replica_concat(hidden1, strategy)
    hidden2_large = tpu_cross_replica_concat(hidden2, strategy)

    enlarged_batch_size = tf.shape(hidden1_large)[0]
    replica_context = tf.distribute.get_replica_context()
    replica_id = tf.cast(
        tf.cast(replica_context.replica_id_in_sync_group, tf.uint32), tf.int32)
    labels_idx = tf.range(batch_size) + replica_id * batch_size
    masks = tf.one_hot(labels_idx, enlarged_batch_size)
    # assuming uniform sampling, MI upper bound == H(X) == H(Y) == -log 1/K
    if double_batch_trick:
      labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
      mi_upper_bound = tf.math.log(tf.cast(enlarged_batch_size*2-1, tf.float32))
    else:
      labels = tf.one_hot(labels_idx, enlarged_batch_size)
      mi_upper_bound = tf.math.log(tf.cast(enlarged_batch_size, tf.float32))
  else:
    hidden1_large = hidden1
    hidden2_large = hidden2
    labels_idx = tf.range(batch_size)
    masks = tf.one_hot(labels_idx, batch_size)
    if double_batch_trick:
      labels = tf.one_hot(labels_idx, batch_size * 2)
      mi_upper_bound = tf.math.log(tf.cast(batch_size*2-1, tf.float32))
    else:
      labels = tf.one_hot(labels_idx, batch_size)
      mi_upper_bound = tf.math.log(tf.cast(batch_size, tf.float32))

  # e_zx: [B, (Z)]
  e_zx = tfd.VonMisesFisher(hidden1, kappa_e)  # scale needs to be batched
  b_zy = tfd.VonMisesFisher(hidden2, kappa_b)
  # b_zy_large: [Bex, (Z)]
  b_zy_large = tfd.VonMisesFisher(hidden2_large, kappa_b)

  # Reversed distributions for bidirectional learning and additional negatives
  if bidirectional or double_batch_trick:
    # [B, (Z)]
    e2_zx = tfd.VonMisesFisher(hidden1, kappa_b)  # scale needs to be batched
    b2_zy = tfd.VonMisesFisher(hidden2, kappa_e)
    # [Bex, (Z)]
    e2_zx_large = tfd.VonMisesFisher(hidden1_large, kappa_b)

  metrics = {}
  # X -> Y
  # zx: [B, Z]
  if sampling:
    zx = e_zx.sample()
  else:
    zx = e_zx.mean_direction
  log_e_zx_x = e_zx.log_prob(zx)
  log_b_zx_y = b_zy.log_prob(zx)
  i_xzx_y = log_e_zx_x - log_b_zx_y  # [B,], residual information I(X;Z|Y)
  # logits_ab: [B, Bex], zx -> [B, 1, Z]
  logits_ab = b_zy_large.log_prob(zx[:, None, :])
  if double_batch_trick:
    # logits_aa: [B, Bex], zx -> [B, 1, Z]
    logits_aa = e2_zx_large.log_prob(zx[:, None, :])
    logits_aa = logits_aa - masks * LARGE_NUM  # Mask out itself
    logits_ab = tf.concat([logits_ab, logits_aa], -1)

  # original_loss_a = -log p(y|zx) -> H(Y|Zx)
  cat_dist_ab = tfd.Categorical(logits=logits_ab)
  h_y_zx = -cat_dist_ab.log_prob(labels_idx)
  i_y_zx = mi_upper_bound - h_y_zx
  loss = beta * i_xzx_y - i_y_zx
  metrics['i_xzx_y'] = tf.reduce_mean(i_xzx_y)
  metrics['i_y_zx'] = tf.reduce_mean(i_y_zx)
  metrics['h_e_zx_x'] = tf.reduce_mean(-log_e_zx_x)
  metrics['h_b_zx_y'] = tf.reduce_mean(-log_b_zx_y)

  # Y -> X
  if bidirectional:
    if sampling:
      zy = b2_zy.sample()
    else:
      zy = b2_zy.mean_direction
    log_b2_zy_y = b2_zy.log_prob(zy)
    log_e2_zy_x = e2_zx.log_prob(zy)
    i_yzy_x = log_b2_zy_y - log_e2_zy_x  # [B,], residual information I(Y;Z|X)

    # logits_ba: [B, Bex], zy -> [B, 1, Z]
    logits_ba = e2_zx_large.log_prob(zy[:, None, :])
    if double_batch_trick:
      # logits_bb: [B, Bex], zy -> [B, 1, Z]
      logits_bb = b_zy_large.log_prob(zy[:, None, :])
      logits_bb = logits_bb - masks * LARGE_NUM  # Mask out itself
      logits_ba = tf.concat([logits_ba, logits_bb], -1)

    # original_loss_b = -log p(x|zy) -> H(X|Zy)
    cat_dist_ba = tfd.Categorical(logits=logits_ba)
    h_x_zy = -cat_dist_ba.log_prob(labels_idx)
    i_x_zy = mi_upper_bound - h_x_zy
    loss += beta * i_yzy_x - i_x_zy
    metrics['i_yzy_x'] = tf.reduce_mean(i_yzy_x)
    metrics['i_x_zy'] = tf.reduce_mean(i_x_zy)

  loss = tf.reduce_mean(loss)
  return loss, logits_ab, labels, metrics
