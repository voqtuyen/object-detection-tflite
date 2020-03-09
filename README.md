# object-detection-tflite



## Convert ssd-based object detection model to tflite

### Convert frozen graph to Tensorflow lite flatbuffer format via command line

```bash
tflite_convert --graph_def_file=tflite_graph.pb \
--output_file=fdetect.tflite \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--input_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=FLOAT \
--allow_custom_ops

```

## References
[1] https://www.tensorflow.org/lite/performance/post_training_quantization
