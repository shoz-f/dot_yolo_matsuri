defmodule YOLOv5 do
  @moduledoc """
  Original work:
    YOLOv5 - https://github.com/ultralytics/yolov5
  """

  alias OnnxInterp, as: NNInterp
  use NNInterp, label: "./model_bank/coco.label",
    model: "./model_bank/yolov5s.onnx",
    url: "https://drive.google.com/uc?authuser=0&export=download&confirm=t&id=1685GlkjJfBVx1yw_U0wdLwnpUNSd6Kom"

  @width  640
  @height 640

  def apply(img) do
    # preprocess
    input0 = img
      |> CImg.resize({@width, @height})
      |> CImg.to_binary([{:range, {0.0, 1.0}}, :nchw])

    # prediction
    output0 = __MODULE__
      |> NNInterp.set_input_tensor(0, input0)
      |> NNInterp.invoke()
      |> NNInterp.get_output_tensor(0)
      |> Nx.from_binary(:f32) |> Nx.reshape({:auto, 85})

    # postprocess
    boxes  = extract_boxes(output0)
    scores = extract_scores(output0)

    NNInterp.non_max_suppression_multi_class(__MODULE__,
      Nx.shape(scores), Nx.to_binary(boxes), Nx.to_binary(scores)
    )
  end

  defp extract_boxes(t), do:
    Nx.slice_along_axis(t, 0, 4, axis: 1) |> Nx.divide(Nx.tensor([@width, @height, @width, @height]))

  defp extract_scores(t), do:
    Nx.multiply(Nx.slice_along_axis(t, 4, 1, axis: 1), Nx.slice_along_axis(t, 5, 80, axis: 1))
end
