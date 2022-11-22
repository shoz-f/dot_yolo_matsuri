defmodule YOLOv7 do
  alias OnnxInterp, as: NNInterp
  use NNInterp,
    model: "./model_bank/yolov7.onnx",
    url: "https://drive.google.com/uc?authuser=0&export=download&confirm=t&id=1T5jP4UZ3Aona4bhv_uGAf91cuHieEPE7"

  @label  (for item <- File.stream!("./model_bank/coco.label") do String.trim_trailing(item) end)
           |> Enum.with_index(&{&2, &1})
           |> Enum.into(%{})

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
      |> Nx.from_binary(:f32) |> Nx.reshape({:auto, 7})

    # postprocess
    {:ok, reformat(output0)}
  end

  def reformat(t) do
    n = Nx.axis_size(t, 0)
    Enum.reduce(0..(n-1), %{}, fn i, map ->
      [_, x1, y1, x2, y2, item, score] = Nx.to_flat_list(t[i])
      label = @label[round(item)]
      box   = [score, x1/@width, y1/@height, x2/@width, y2/@height, i]
      Map.update(map, label, [box], fn boxes -> [box|boxes] end)
    end)
  end
end
