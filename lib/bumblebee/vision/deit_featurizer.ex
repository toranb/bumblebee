defmodule Bumblebee.Vision.DeitFeaturizer do
  alias Bumblebee.Shared

  options = [
    resize: [
      default: true,
      doc: "whether to resize the input to the given `:size`"
    ],
    size: [
      default: %{height: 256, width: 256},
      doc: """
      the size to resize the input to, given as `%{height: ..., width: ...}`. Only has
      an effect if `:resize` is `true`
      """
    ],
    resize_method: [
      default: :bicubic,
      doc:
        "the resizing method, either of `:nearest`, `:bilinear`, `:bicubic`, `:lanczos3`, `:lanczos5`"
    ],
    center_crop: [
      default: true,
      doc: """
      whether to crop the input at the center. If the input size is smaller than `:crop_size` along
      any edge, the image is padded with zeros and then center cropped
      """
    ],
    crop_size: [
      default: %{height: 224, width: 224},
      doc: """
      the size to center crop the image to, given as `%{height: ..., width: ...}`. Only has an effect
      if `:center_crop` is `true`
      """
    ],
    normalize: [
      default: true,
      doc: "whether or not to normalize the input with mean and standard deviation"
    ],
    image_mean: [
      default: [0.485, 0.456, 0.406],
      doc: "the sequence of mean values for each channel, to be used when normalizing images"
    ],
    image_std: [
      default: [0.229, 0.224, 0.225],
      doc:
        "the sequence of standard deviations for each channel, to be used when normalizing images"
    ]
  ]

  @moduledoc """
  DeiT featurizer for image data.

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
      images =
        image
        |> Image.to_batched_tensor()
        |> Nx.as_type(:f32)
        |> Image.normalize_channels(length(featurizer.image_mean))

      if featurizer.resize do
        %{height: height, width: width} = featurizer.size
        NxImage.resize(images, {height, width}, method: featurizer.resize_method)
      else
        images
      end
    end
    |> Nx.concatenate()
  end

  @impl true
  def batch_template(featurizer, batch_size) do
    %{height: height, width: width} = featurizer.size
    num_channels = length(featurizer.image_mean)
    Nx.template({batch_size, height, width, num_channels}, :f32)
  end

  @impl true
  def process_batch(featurizer, images) do
    images =
      if featurizer.center_crop do
        %{height: height, width: width} = featurizer.crop_size
        NxImage.center_crop(images, {height, width})
      else
        images
      end

    images = NxImage.to_continuous(images, 0, 1)

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
          size: {"size", image_size()},
          resize_method: {"resample", resize_method()},
          center_crop: {"do_center_crop", boolean()},
          crop_size: {"crop_size", image_size()},
          normalize: {"do_normalize", boolean()},
          image_mean: {"image_mean", list(number())},
          image_std: {"image_std", list(number())}
        )

      @for.config(featurizer, opts)
    end
  end
end
