# Convert ssd-based object detection models to tflite models



## Train object detection models with tensorflow object detection api

### Installation & running locally

Please refer to tensorflow docs: [installation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) and [running locally](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md)

### Notes

- However, you should get the config files from [samples](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs). Otherwise, the resulted tflite model will produce random outputs


## Convert ssd-based object detection model to tflite

### 1. Get tensorflow frozen graph with compatible ops used with TensorFlow Lite
We start with a checkpoint and get a TensorFlow frozen graph with compatible ops that we can use with TensorFlow Lite. To get the frozen graph, run the export_tflite_ssd_graph.py script from the models/research directory with this command

```bash
python object_detection/export_tflite_ssd_graph.py \
--pipeline_config_path=$CONFIG_FILE \
--trained_checkpoint_prefix=$CHECKPOINT_PATH \
--output_directory=$OUTPUT_DIR \
--add_postprocessing_op=true
```

### 2. Convert frozen graph to Tensorflow lite flatbuffer format via command line

```bash
tflite_convert --output_file=fdetect.tflite \
--graph_def_file=tflite_graph.pb \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=FLOAT \
--allow_custom_ops

```

## References
[1] https://www.tensorflow.org/lite/performance/post_training_quantization  
[2] https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
