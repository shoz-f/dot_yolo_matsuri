defmodule YOLOs do
  @palette CImg.Util.rand_palette("./model_bank/coco.label")

  def run_all(path) do
    img = CImg.load(path)

    Enum.each(YOLOs.Application.yolos(), fn yolo ->
      with {:ok, res} = yolo.apply(img) do
        Enum.reduce(res, CImg.builder(img), &draw_item(&1, &2))
        |> CImg.save("#{yolo}.jpg")
      end
    end)
  end

  def run(yolo, path) do
    img = CImg.load(path)

    with {:ok, res} = yolo.apply(img) do
      Enum.reduce(res, CImg.builder(img), &draw_item(&1, &2))
      |> CImg.save("#{yolo}.jpg")
    end
  end

  defp draw_item({item, boxes}, canvas) do
    color = @palette[item]
    Enum.reduce(boxes, canvas, fn [_score, x1, y1, x2, y2, _index], canvas ->
      x1 = if x1 < 0.0, do: 0.0, else: x1
      y1 = if y1 < 0.0, do: 0.0, else: y1
      x2 = if x2 > 1.0, do: 1.0, else: x2
      y2 = if y2 > 1.0, do: 1.0, else: y2

      CImg.fill_rect(canvas, x1, y1, x2, y2, color, 0.35)
    end)
  end
end
