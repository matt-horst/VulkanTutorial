struct ShaderData {
    float4x4 model;
    float4x4 view;
    float4x4 projection;
};

struct VertexInput {
  float2 position;
  float3 color;
}

struct VertexOutput {
  float4 vs_position : SV_Position;
  float3 color;
};

ConstantBuffer<ShaderData> ubo;

[shader("vertex")]
VertexOutput vertMain(VertexInput vs_in) {
  VertexOutput output;

  output.vs_position = mul(ubo.projection, mul(ubo.view, mul(ubo.model, float4(vs_in.position, 0.0, 1.0))));
  output.color = vs_in.color;

  return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput fs_in) : SV_Target {
  return float4(fs_in.color, 1.0);
}
