defmodule Bumblebee.Text.JinaBertV2Test do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  @moduletag model_test_tags()

  test "text_embedding serving" do
    repo = {:hf, "jinaai/jina-embeddings-v2-base-code"}

    {:ok, %{model: model, params: params, spec: spec} = model_info} =
      Bumblebee.load_model(repo,
        params_filename: "model.safetensors",
        spec_overrides: [architecture: :base]
      )

    {:ok, tokenizer} = Bumblebee.load_tokenizer(repo)

    serving =
      Bumblebee.Text.TextEmbedding.text_embedding(model_info, tokenizer,
        compile: [batch_size: 2, sequence_length: 512],
        defn_options: [compiler: EXLA],
        output_attribute: :hidden_state,
        output_pool: :mean_pooling
      )

    %{embedding: embedding} = Nx.Serving.run(serving, "How is the weather today?")

    assert Nx.all_close(Nx.tensor([1.0732901, -0.2474489, 0.52255607]), embedding[0..2])
           |> Nx.to_number() == 1
  end
end
