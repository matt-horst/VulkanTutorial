static float2 positions[3] = float2[](float2(0.0, -0.5), float2(0.5, 0.5), float2(-0.5, 0.5));
static float3 colors[3] = float3[](float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0), float3(0.0, 0.0, 1.0));

struct VertexOutput {
  float4 vs_position : SV_Position;
  float3 color;
};

[shader("vertex")]
VertexOutput vertMain(uint vid: SV_VertexID) {
  VertexOutput output;

  output.vs_position = float4(positions[vid], 0.0, 1.0);
  output.color = colors[vid];

  return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput fs_in) : SV_Target {
  return float4(fs_in.color, 1.0);
}
