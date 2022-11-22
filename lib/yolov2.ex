defmodule YOLOv2 do
  alias OnnxInterp, as: NNInterp
  use NNInterp, label: "./model_bank/coco.label",
    model: "./model_bank/yolov2-coco-9.onnx",
    url: "https://media.githubusercontent.com/media/onnx/models/main/vision/object_detection_segmentation/yolov2-coco/model/yolov2-coco-9.onnx"

  @width   416
  @height  416

  def apply(img) do
    # preprocess
    input0 = img
      |> CImg.resize({@width, @height})
      |> CImg.to_binary([:nchw])

    # prediction
    output0 = __MODULE__
      |> NNInterp.set_input_tensor(0, input0)
      |> NNInterp.invoke()
      |> NNInterp.get_output_tensor(0)
      |> Nx.from_binary(:f32) |> Nx.reshape({5, 85, 13, 13})

    # postprocess
    output0 = Nx.transpose(output0, axes: [1, 0, 2, 3]) |> Nx.reshape({85, :auto})
      # output0 => [box(4),box_score(1),class_score(80)]x[anchor0[13x13],anchor1[13x13],..,anchor4[13x13]]

    boxes  = extract_boxes(output0)
    scores = extract_scores(output0)

    NNInterp.non_max_suppression_multi_class(__MODULE__,
      Nx.shape(scores), Nx.to_binary(boxes), Nx.to_binary(scores)
    )
  end

  @grid     PostDNN.meshgrid({@width, @height}, 32, [:transpose, :normalize]) |> Nx.tile([5])
  @anchors Nx.tensor([[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]])

  defp extract_boxes(t) do
    # decode box center coordinate on {1.0, 1.0}
    center = Nx.logistic(t[0..1])
      |> Nx.multiply(@grid[2..3])  # * pitch(x,y)
      |> Nx.add(@grid[0..1])    # + grid(x,y)
      |> Nx.transpose()

    # decode box size
    size = Nx.exp(t[2..3])
      |> Nx.multiply(@grid[2..3]) # * pitch(x,y)
      # multiply @anchors
      |> Nx.reshape({2, 5, :auto})
      |> Nx.transpose(axes: [2, 1, 0])
      |> Nx.multiply(@anchors)
      # get a transposed box sizes.
      |> Nx.transpose(axes: [1, 0, 2])
      |> Nx.reshape({:auto, 2})

    Nx.concatenate([center, size], axis: 1)
  end

  defp extract_scores(t) do
    # decode box confidence
    confidence = Nx.logistic(t[4])

    # decode class scores: (softmax normalized class score)*(box confidence)
    exp = Nx.exp(t[5..-1//1])

    Nx.divide(exp, Nx.sum(exp, axes: [0])) # apply softmax on each class score
    |> Nx.multiply(confidence)
    |> Nx.transpose()
  end
end
