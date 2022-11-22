defmodule YOLOX do
  alias OnnxInterp, as: NNInterp
  use NNInterp, label: "./model_bank/coco.label",
    model: "./model_bank/yolox_s.onnx",
    url: "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"

  @width  640
  @height 640

  def apply(img) do
    # preprocess
    input0 = img
#      |> CImg.resize({@width, @height}, :ul, 114)
      |> CImg.resize({@width, @height})
      |> CImg.to_binary([{:range, {0.0, 255.0}}, :nchw])

    # prediction
    output0 = __MODULE__
      |> NNInterp.set_input_tensor(0, input0)
      |> NNInterp.invoke()
      |> NNInterp.get_output_tensor(0)
      |> Nx.from_binary(:f32) |> Nx.reshape({:auto, 85})

    # postprocess
    output0 = Nx.transpose(output0)

    boxes  = extract_boxes(output0)
    scores = extract_scores(output0)

    NNInterp.non_max_suppression_multi_class(__MODULE__,
      Nx.shape(scores), Nx.to_binary(boxes), Nx.to_binary(scores)
    )
    #|> PostDNN.adjust2letterbox(CImg.Util.aspect(img))
  end


  @grid PostDNN.meshgrid({@width, @height}, [8, 16, 32], [:transpose, :normalize])

  defp  extract_boxes(t) do
    # decode box center coordinate on {1.0, 1.0}
    center = t[0..1]
      |> Nx.multiply(@grid[2..3])  # * pitch(x,y)
      |> Nx.add(@grid[0..1])    # + grid(x,y)

    # decode box size
    size = t[2..3]
      |> Nx.exp()
      |> Nx.multiply(@grid[2..3]) # * pitch(x,y)

    Nx.concatenate([center, size]) |> Nx.transpose()
  end

  defp extract_scores(t) do
    Nx.multiply(t[4], t[5..-1//1])
    |> Nx.transpose()
  end
end
