struct ShaderData {
    float4x4 model;
    float4x4 view;
    float4x4 projection;
};

struct VertexInput {
  float2 position;
  float3 color;
  float2 texCoord;
}

struct VertexOutput {
  float4 vs_position : SV_Position;
  float3 color;
  float2 texCoord;
};

ConstantBuffer<ShaderData> ubo;

[shader("vertex")]
VertexOutput vertMain(VertexInput vs_in) {
  VertexOutput output;

  output.vs_position = mul(ubo.projection, mul(ubo.view, mul(ubo.model, float4(vs_in.position, 0.0, 1.0))));
  output.color = vs_in.color;
  output.texCoord = vs_in.texCoord;

  return output;
}


[shader("fragment")]
float4 fragMain(VertexOutput fs_in, uniform Sampler2D texture) : SV_Target {
    return texture.Sample(fs_in.texCoord);
}
