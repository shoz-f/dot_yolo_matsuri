defmodule YOLOv3 do
  @moduledoc """
  Original work:
    tensorflow-yolov4-tflite - https://github.com/hunglc007/tensorflow-yolov4-tflite
  """

  alias OnnxInterp, as: NNInterp
  use NNInterp, label: "./model_bank/coco.label",
    model: "./model_bank/yolov3-416.onnx",
    url: "https://drive.google.com/uc??authuser=0&export=download&confirm=t&id=1NIYVUtWbYEZuhEfvIm5S5R01T0J4F69Q"

  @width  416
  @height 416

  def apply(img) do
    # preprocess
    input0 = img
      |> CImg.resize({@width, @height})
      |> CImg.to_binary(range: {0.0, 1.0})

    # prediction
    output0 = __MODULE__
      |> NNInterp.set_input_tensor(0, input0)
      |> NNInterp.invoke()
      |> NNInterp.get_output_tensor(0)
      |> Nx.from_binary(:f32) |> Nx.reshape({:auto, 84})

    # postprocess
    boxes  = extract_boxes(output0)
    scores = extract_scores(output0)

    NNInterp.non_max_suppression_multi_class(__MODULE__,
      Nx.shape(scores), Nx.to_binary(boxes), Nx.to_binary(scores),
      boxrepr: :corner
    )
  end

  defp extract_boxes(t) do
    Nx.concatenate(
      [
        Nx.slice_along_axis(t, 1, 1, axis: 1),
        Nx.slice_along_axis(t, 0, 1, axis: 1),
        Nx.slice_along_axis(t, 3, 1, axis: 1),
        Nx.slice_along_axis(t, 2, 1, axis: 1)
      ],
      axis: 1
    )
  end

  defp extract_scores(t), do:
    Nx.slice_along_axis(t, 4, 80, axis: 1)
end
