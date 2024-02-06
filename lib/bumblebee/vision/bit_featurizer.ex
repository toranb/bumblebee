defmodule Bumblebee.Vision.BitFeaturizer do
  alias Bumblebee.Shared

  options = [
    resize: [
      default: true,
      doc: "whether to resize the input to the given `:size`"
    ],
    size: [
      default: 224,
      doc: """
      the size to resize the input to. A single number, a `{height, width}` tuple, or a map specifying the shortest edge.
      Only has an effect if `:resize` is `true`
      """
    ],
    resize_method: [
      default: :bicubic,
      doc:
        "the resizing method, either of `:nearest`, `:bilinear`, `:bicubic`, `:lanczos3`, `:lanczos5`"
    ],
    center_crop: [
      default: true,
      doc: "whether to crop the image at the center to given `:crop_size`"
    ],
    crop_size: [
      default: {224, 224},
      doc: """
      the size to crop the input to. A `{height, width}` tuple
      Only has an effect if `:crop` is `true`
      """
    ],
    rescale: [
      default: true,
      doc: "whether to rescale the input by the given `:rescale_factor`"
    ],
    rescale_factor: [
      default: 224,
      doc: """
      the factor by which to rescale the input. A single number
      Only has an effect if `:rescale` is `true`
      """
    ],
    normalize: [
      default: true,
      doc: "whether or not to normalize the input with mean and standard deviation"
    ],
    image_mean: [
      default: [0.5, 0.5, 0.5],
      doc: "the sequence of mean values for each channel, to be used when normalizing images"
    ],
    image_std: [
      default: [0.5, 0.5, 0.5],
      doc:
        "the sequence of standard deviations for each channel, to be used when normalizing images"
    ]
  ]

  @moduledoc """
  BiT featurizer for image data.

  ## Configuration

  #{Shared.options_doc(options)}
  """

  defstruct Shared.option_defaults(options)

  @behaviour Bumblebee.Featurizer
  @behaviour Bumblebee.Configurable

  alias Bumblebee.Utils.Image

  @impl true
  def config(featurizer, opts) do
    Shared.put_config_attrs(featurizer, opts)
  end

  @impl true
  def process_input(featurizer, images) do
    images = List.wrap(images)

    for image <- images do
      image
      |> Image.to_batched_tensor()
      |> Nx.as_type(:f32)
      |> Image.normalize_channels(length(featurizer.image_mean))
      |> maybe_resize(featurizer)
      |> maybe_center_crop(featurizer)
      |> maybe_rescale(featurizer)
    end
    |> Nx.concatenate()
  end

  defp maybe_resize(images, featurizer) do
    if featurizer.resize do
      resize(images, featurizer)
    else
      images
    end
  end

  defp resize(images, featurizer) do
    case featurizer.size do
      %{"shortest_edge" => size} ->
        NxImage.resize_short(images, size, method: featurizer.resize_method)

      _ ->
        size = Image.normalize_size(featurizer.size)
        NxImage.resize(images, size, method: featurizer.resize_method)
    end
  end

  defp maybe_center_crop(images, featurizer) do
    if featurizer.center_crop do
      %{"height" => crop_height, "width" => crop_width} = featurizer.crop_size
      NxImage.center_crop(images, {crop_height, crop_width})
    else
      images
    end
  end

  defp maybe_rescale(images, featurizer) do
    if featurizer.rescale do
      Nx.multiply(images, featurizer.rescale_factor)
    else
      images
    end
  end

  @impl true
  def batch_template(featurizer, batch_size) do
    {height, width} = Image.normalize_size(featurizer.size)
    num_channels = length(featurizer.image_mean)
    Nx.template({batch_size, height, width, num_channels}, :f32)
  end

  @impl true
  def process_batch(featurizer, images) do
    images =
      if featurizer.normalize do
        NxImage.normalize(
          images,
          Nx.tensor(featurizer.image_mean),
          Nx.tensor(featurizer.image_std)
        )
      else
        images
      end

    %{"pixel_values" => images}
  end

  defimpl Bumblebee.HuggingFace.Transformers.Config do
    def load(featurizer, data) do
      import Shared.Converters

      opts =
        convert!(data,
          resize: {"do_resize", boolean()},
          size:
            {"size", one_of([number(), tuple([number(), number()]), map(string(), number())])},
          resize_method: {"resample", resize_method()},
          center_crop: {"do_center_crop", boolean()},
          crop_size: {"crop_size", map(string(), number())},
          rescale: {"do_rescale", boolean()},
          rescale_factor: {"rescale_factor", number()},
          normalize: {"do_normalize", boolean()},
          image_mean: {"image_mean", list(number())},
          image_std: {"image_std", list(number())}
        )

      @for.config(featurizer, opts)
    end
  end
end
