import tensorflow_hub as tfhub

INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'


def get_inception_model(inceptionv3=False):
  if inceptionv3:
    return tfhub.load(
      'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4')
  else:
    return tfhub.load(INCEPTION_TFHUB)

get_inception_model(False)
