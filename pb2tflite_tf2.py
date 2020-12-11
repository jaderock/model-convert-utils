import os
import argparse
import tensorflow as tf

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Convert tensorflow 2 .pb model to tflite model.')
  parser.add_argument('--pb_dir', type=str, default=None, help='path to tensorflow 2 .pb model directory')
  parser.add_argument('--tflite_file', type=str, default=None, help='file path to tflite model')
  args = parser.parse_args()

  if (args.pb_dir is None) or (args.tflite_file is None):
    print("Error: must specify pb_dir and tflite_file")
    quit()

  tflite_folder_path = os.path.dirname(args.tflite_file)
  if not os.path.exists(tflite_folder_path):
    os.makedirs(tflite_folder_path)

  # Convert the model
  converter = tf.lite.TFLiteConverter.from_saved_model(args.pb_dir) # path to the SavedModel directory
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
  tflite_model = converter.convert()

  # Save the model.
  with open(args.tflite_file, 'wb') as f:
    f.write(tflite_model)

