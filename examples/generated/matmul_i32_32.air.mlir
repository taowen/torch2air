// -----// IR Dump After DmaToChannel (air-dma-to-channel) ('builtin.module' operation) //----- //
#config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 0], [0, 0, 1], [1, 1, 0]]>
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d2, d4, d5, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [8, 4, 32], transposePackIndices = [0, 1], unpackEmpty = [false, false], innerPerm = [[0, 1], [1, 0]], outerPerm = [[0, 1], [1, 0]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 7 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 7 >= 0)>
#set2 = affine_set<()[s0, s1] : (s0 - 2 == 0, s1 >= 0, -s1 + 7 >= 0)>
#set3 = affine_set<()[s0, s1] : (s0 - 3 == 0, s1 >= 0, -s1 + 7 >= 0)>
#set4 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 == 0)>
#set5 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 1 == 0)>
#set6 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 2 == 0)>
#set7 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 3 == 0)>
#set8 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 4 == 0)>
#set9 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 5 == 0)>
#set10 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 6 == 0)>
#set11 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 7 == 0)>
#translation = #iree_codegen.translation_info<pipeline = Custom>
#device_target_xrt_lite = #hal.device.target<"xrt-lite", [#executable_target_amdaie_pdi_fb]> : !hal.device
module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #device_target_xrt_lite
  hal.executable private @forward_dispatch_0 {
    hal.executable.variant public @amdaie_pdi_fb target(#executable_target_amdaie_pdi_fb) {
      hal.executable.export public @forward_dispatch_0_matmul_32x32x32_i32 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
        %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        air.channel @channel_0 [1, 1]
        air.channel @channel_1 [1, 1]
        air.channel @channel_2 [1, 1] {broadcast_shape = [1, 8]}
        air.channel @channel_3 [1, 1] {broadcast_shape = [1, 8]}
        air.channel @channel_4 [1, 1] {broadcast_shape = [1, 8]}
        air.channel @channel_5 [1, 1] {broadcast_shape = [1, 8]}
        air.channel @channel_6 [1, 1] {broadcast_shape = [4, 1]}
        air.channel @channel_7 [1, 1] {broadcast_shape = [4, 1]}
        air.channel @channel_8 [1, 1] {broadcast_shape = [4, 1]}
        air.channel @channel_9 [1, 1] {broadcast_shape = [4, 1]}
        air.channel @channel_10 [1, 1] {broadcast_shape = [4, 1]}
        air.channel @channel_11 [1, 1] {broadcast_shape = [4, 1]}
        air.channel @channel_12 [1, 1] {broadcast_shape = [4, 1]}
        air.channel @channel_13 [1, 1] {broadcast_shape = [4, 1]}
        air.channel @channel_14 [4, 8]
        air.channel @channel_15 [1, 1]
        func.func @forward_dispatch_0_matmul_32x32x32_i32(%arg0: memref<32x32xi32>, %arg1: memref<32x32xi32>, %arg2: memref<32x32xi32>) attributes {translation_info = #translation} {
          %c1 = arith.constant 1 : index
          %0 = air.launch async (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<32x32xi32>, memref<32x32xi32>, memref<32x32xi32> attributes {id = 3 : i32} {
            %c1024 = arith.constant 1024 : index
            %c1_0 = arith.constant 1 : index
            %c32 = arith.constant 32 : index
            %c256 = arith.constant 256 : index
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c4 = arith.constant 4 : index
            %1 = air.wait_all async 
            %2 = air.wait_all async 
            %3 = air.wait_all async 
            %4 = air.wait_all async 
            %5 = air.channel.put async  @channel_0[] (%arg7[%c0, %c0, %c0, %c0] [%c4, %c1_0, %c8, %c32] [%c256, %c32, %c32, %c1_0]) : (memref<32x32xi32>)
            %6 = air.wait_all async 
            %7 = air.wait_all async 
            %8 = air.wait_all async 
            %9 = air.wait_all async 
            %10 = air.wait_all async 
            %11 = air.wait_all async 
            %12 = air.wait_all async 
            %13 = air.wait_all async 
            %14 = air.wait_all async 
            %15 = air.wait_all async 
            %16 = air.wait_all async 
            %17 = air.wait_all async 
            %18 = air.wait_all async 
            %19 = air.wait_all async 
            %20 = air.wait_all async 
            %21 = air.wait_all async 
            %22 = air.wait_all async 
            %23 = air.wait_all async 
            %24 = air.wait_all async 
            %25 = air.wait_all async 
            %26 = air.wait_all async 
            %27 = air.wait_all async 
            %28 = air.wait_all async 
            %29 = air.wait_all async 
            %30 = air.wait_all async 
            %31 = air.wait_all async 
            %32 = air.wait_all async 
            %33 = air.wait_all async 
            %34 = air.wait_all async 
            %35 = air.wait_all async 
            %36 = air.wait_all async 
            %37 = air.wait_all async 
            %38 = air.wait_all async 
            %39 = air.wait_all async 
            %40 = air.wait_all async 
            %41 = air.wait_all async 
            %42 = air.wait_all async 
            %43 = air.wait_all async 
            %44 = air.wait_all async 
            %45 = air.wait_all async 
            %46 = air.wait_all async 
            %47 = air.wait_all async 
            %48 = air.wait_all async 
            %49 = air.wait_all async 
            %50 = air.wait_all async 
            %51 = air.wait_all async 
            %52 = air.wait_all async 
            %53 = air.wait_all async 
            %54 = air.wait_all async 
            %55 = air.wait_all async 
            %56 = air.wait_all async 
            %57 = air.wait_all async 
            %58 = air.wait_all async 
            %59 = air.wait_all async 
            %60 = air.wait_all async 
            %61 = air.wait_all async 
            %62 = air.wait_all async 
            %63 = air.wait_all async 
            %64 = air.wait_all async 
            %65 = air.wait_all async 
            %66 = air.wait_all async 
            %67 = air.wait_all async 
            %68 = air.wait_all async 
            %69 = air.wait_all async 
            %70 = air.wait_all async 
            %71 = air.wait_all async 
            %72 = air.wait_all async 
            %73 = air.wait_all async 
            %74 = air.wait_all async 
            %75 = air.wait_all async 
            %76 = air.wait_all async 
            %77 = air.wait_all async 
            %78 = air.wait_all async 
            %79 = air.wait_all async 
            %80 = air.wait_all async 
            %81 = air.wait_all async 
            %82 = air.wait_all async 
            %83 = air.wait_all async 
            %84 = air.wait_all async 
            %85 = air.wait_all async 
            %86 = air.wait_all async 
            %87 = air.wait_all async 
            %88 = air.wait_all async 
            %89 = air.wait_all async 
            %90 = air.wait_all async 
            %91 = air.wait_all async 
            %92 = air.wait_all async 
            %93 = air.wait_all async 
            %94 = air.wait_all async 
            %95 = air.wait_all async 
            %96 = air.wait_all async 
            %97 = air.wait_all async 
            %98 = air.wait_all async 
            %99 = air.wait_all async 
            %100 = air.wait_all async 
            %101 = air.wait_all async 
            %102 = air.wait_all async 
            %103 = air.wait_all async 
            %104 = air.wait_all async 
            %105 = air.wait_all async 
            %106 = air.wait_all async 
            %107 = air.wait_all async 
            %108 = air.wait_all async 
            %109 = air.wait_all async 
            %110 = air.wait_all async 
            %111 = air.wait_all async 
            %112 = air.wait_all async 
            %113 = air.wait_all async 
            %114 = air.wait_all async 
            %115 = air.wait_all async 
            %116 = air.wait_all async 
            %117 = air.wait_all async 
            %118 = air.wait_all async 
            %119 = air.wait_all async 
            %120 = air.wait_all async 
            %121 = air.wait_all async 
            %122 = air.wait_all async 
            %123 = air.wait_all async 
            %124 = air.wait_all async 
            %125 = air.wait_all async 
            %126 = air.wait_all async 
            %127 = air.wait_all async 
            %128 = air.wait_all async 
            %129 = air.wait_all async 
            %130 = air.wait_all async 
            %131 = air.wait_all async 
            %132 = air.wait_all async 
            %133 = air.wait_all async 
            %134 = air.wait_all async 
            %135 = air.wait_all async 
            %136 = air.wait_all async 
            %137 = air.wait_all async 
            %138 = air.wait_all async 
            %139 = air.wait_all async 
            %140 = air.wait_all async 
            %141 = air.wait_all async 
            %142 = air.wait_all async 
            %143 = air.wait_all async 
            %144 = air.wait_all async 
            %145 = air.wait_all async 
            %146 = air.wait_all async 
            %147 = air.wait_all async 
            %148 = air.wait_all async 
            %149 = air.wait_all async 
            %150 = air.wait_all async 
            %151 = air.wait_all async 
            %152 = air.wait_all async 
            %153 = air.wait_all async 
            %154 = air.wait_all async 
            %155 = air.wait_all async 
            %156 = air.wait_all async 
            %157 = air.wait_all async 
            %158 = air.wait_all async 
            %159 = air.wait_all async 
            %160 = air.wait_all async 
            %161 = air.wait_all async 
            %162 = air.wait_all async 
            %163 = air.wait_all async 
            %164 = air.wait_all async 
            %165 = air.wait_all async 
            %166 = air.wait_all async 
            %167 = air.wait_all async 
            %168 = air.wait_all async 
            %169 = air.wait_all async 
            %170 = air.wait_all async 
            %171 = air.wait_all async 
            %172 = air.wait_all async 
            %173 = air.wait_all async 
            %174 = air.wait_all async 
            %175 = air.wait_all async 
            %176 = air.wait_all async 
            %177 = air.wait_all async 
            %178 = air.wait_all async 
            %179 = air.wait_all async 
            %180 = air.wait_all async 
            %181 = air.wait_all async 
            %182 = air.wait_all async 
            %183 = air.wait_all async 
            %184 = air.wait_all async 
            %185 = air.wait_all async 
            %186 = air.wait_all async 
            %187 = air.wait_all async 
            %188 = air.wait_all async 
            %189 = air.wait_all async 
            %190 = air.wait_all async 
            %191 = air.wait_all async 
            %192 = air.wait_all async 
            %193 = air.wait_all async 
            %194 = air.wait_all async 
            %195 = air.wait_all async 
            %196 = air.wait_all async 
            %197 = air.wait_all async 
            %198 = air.wait_all async 
            %199 = air.wait_all async 
            %200 = air.wait_all async 
            %201 = air.wait_all async 
            %202 = air.wait_all async 
            %203 = air.wait_all async 
            %204 = air.wait_all async 
            %205 = air.wait_all async 
            %206 = air.wait_all async 
            %207 = air.wait_all async 
            %208 = air.wait_all async 
            %209 = air.wait_all async 
            %210 = air.wait_all async 
            %211 = air.wait_all async 
            %212 = air.wait_all async 
            %213 = air.wait_all async 
            %214 = air.wait_all async 
            %215 = air.wait_all async 
            %216 = air.wait_all async 
            %217 = air.wait_all async 
            %218 = air.wait_all async 
            %219 = air.wait_all async 
            %220 = air.wait_all async 
            %221 = air.wait_all async 
            %222 = air.wait_all async 
            %223 = air.wait_all async 
            %224 = air.wait_all async 
            %225 = air.wait_all async 
            %226 = air.wait_all async 
            %227 = air.wait_all async 
            %228 = air.wait_all async 
            %229 = air.wait_all async 
            %230 = air.wait_all async 
            %231 = air.wait_all async 
            %232 = air.wait_all async 
            %233 = air.wait_all async 
            %234 = air.wait_all async 
            %235 = air.wait_all async 
            %236 = air.wait_all async 
            %237 = air.wait_all async 
            %238 = air.wait_all async 
            %239 = air.wait_all async 
            %240 = air.wait_all async 
            %241 = air.wait_all async 
            %242 = air.wait_all async 
            %243 = air.wait_all async 
            %244 = air.wait_all async 
            %245 = air.wait_all async 
            %246 = air.wait_all async 
            %247 = air.wait_all async 
            %248 = air.wait_all async 
            %249 = air.wait_all async 
            %250 = air.wait_all async 
            %251 = air.wait_all async 
            %252 = air.wait_all async 
            %253 = air.wait_all async 
            %254 = air.wait_all async 
            %255 = air.wait_all async 
            %256 = air.wait_all async 
            %257 = air.wait_all async 
            %258 = air.channel.put async  @channel_1[] (%arg8[%c0, %c0, %c0, %c0] [%c8, %c1_0, %c32, %c4] [%c4, %c1024, %c32, %c1_0]) : (memref<32x32xi32>)
            %259 = air.wait_all async 
            %260 = air.wait_all async 
            %261 = air.wait_all async 
            %262 = air.wait_all async 
            %263 = air.wait_all async 
            %264 = air.wait_all async 
            %265 = air.wait_all async 
            %266 = air.wait_all async 
            %267 = air.wait_all async 
            %268 = air.wait_all async 
            %269 = air.wait_all async 
            %270 = air.wait_all async 
            %271 = air.wait_all async 
            %272 = air.wait_all async 
            %273 = air.wait_all async 
            %274 = air.wait_all async 
            %275 = air.wait_all async 
            %276 = air.wait_all async 
            %277 = air.wait_all async 
            %278 = air.wait_all async 
            %279 = air.wait_all async 
            %280 = air.wait_all async 
            %281 = air.wait_all async 
            %282 = air.wait_all async 
            %283 = air.wait_all async 
            %284 = air.wait_all async 
            %285 = air.wait_all async 
            %286 = air.wait_all async 
            %287 = air.wait_all async 
            %288 = air.wait_all async 
            %289 = air.wait_all async 
            %290 = air.wait_all async 
            %291 = air.wait_all async 
            %292 = air.wait_all async 
            %293 = air.wait_all async 
            %294 = air.wait_all async 
            %295 = air.wait_all async 
            %296 = air.wait_all async 
            %297 = air.wait_all async 
            %298 = air.wait_all async 
            %299 = air.wait_all async 
            %300 = air.wait_all async 
            %301 = air.wait_all async 
            %302 = air.wait_all async 
            %303 = air.wait_all async 
            %304 = air.wait_all async 
            %305 = air.wait_all async 
            %306 = air.wait_all async 
            %307 = air.wait_all async 
            %308 = air.wait_all async 
            %309 = air.wait_all async 
            %310 = air.wait_all async 
            %311 = air.wait_all async 
            %312 = air.wait_all async 
            %313 = air.wait_all async 
            %314 = air.wait_all async 
            %315 = air.wait_all async 
            %316 = air.wait_all async 
            %317 = air.wait_all async 
            %318 = air.wait_all async 
            %319 = air.wait_all async 
            %320 = air.wait_all async 
            %321 = air.wait_all async 
            %322 = air.wait_all async 
            %323 = air.wait_all async 
            %324 = air.wait_all async 
            %325 = air.wait_all async 
            %326 = air.wait_all async 
            %327 = air.wait_all async 
            %328 = air.wait_all async 
            %329 = air.wait_all async 
            %330 = air.wait_all async 
            %331 = air.wait_all async 
            %332 = air.wait_all async 
            %333 = air.wait_all async 
            %334 = air.wait_all async 
            %335 = air.wait_all async 
            %336 = air.wait_all async 
            %337 = air.wait_all async 
            %338 = air.wait_all async 
            %339 = air.wait_all async 
            %340 = air.wait_all async 
            %341 = air.wait_all async 
            %342 = air.wait_all async 
            %343 = air.wait_all async 
            %344 = air.wait_all async 
            %345 = air.wait_all async 
            %346 = air.wait_all async 
            %347 = air.wait_all async 
            %348 = air.wait_all async 
            %349 = air.wait_all async 
            %350 = air.wait_all async 
            %351 = air.wait_all async 
            %352 = air.wait_all async 
            %353 = air.wait_all async 
            %354 = air.wait_all async 
            %355 = air.wait_all async 
            %356 = air.wait_all async 
            %357 = air.wait_all async 
            %358 = air.wait_all async 
            %359 = air.wait_all async 
            %360 = air.wait_all async 
            %361 = air.wait_all async 
            %362 = air.wait_all async 
            %363 = air.wait_all async 
            %364 = air.wait_all async 
            %365 = air.wait_all async 
            %366 = air.wait_all async 
            %367 = air.wait_all async 
            %368 = air.wait_all async 
            %369 = air.wait_all async 
            %370 = air.wait_all async 
            %371 = air.wait_all async 
            %372 = air.wait_all async 
            %373 = air.wait_all async 
            %374 = air.wait_all async 
            %375 = air.wait_all async 
            %376 = air.wait_all async 
            %377 = air.wait_all async 
            %378 = air.wait_all async 
            %379 = air.wait_all async 
            %380 = air.wait_all async 
            %381 = air.wait_all async 
            %382 = air.wait_all async 
            %383 = air.wait_all async 
            %384 = air.wait_all async 
            %385 = air.wait_all async 
            %386 = air.wait_all async 
            %387 = air.wait_all async 
            %388 = air.wait_all async 
            %389 = air.wait_all async 
            %390 = air.wait_all async 
            %391 = air.wait_all async 
            %392 = air.wait_all async 
            %393 = air.wait_all async 
            %394 = air.wait_all async 
            %395 = air.wait_all async 
            %396 = air.wait_all async 
            %397 = air.wait_all async 
            %398 = air.wait_all async 
            %399 = air.wait_all async 
            %400 = air.wait_all async 
            %401 = air.wait_all async 
            %402 = air.wait_all async 
            %403 = air.wait_all async 
            %404 = air.wait_all async 
            %405 = air.wait_all async 
            %406 = air.wait_all async 
            %407 = air.wait_all async 
            %408 = air.wait_all async 
            %409 = air.wait_all async 
            %410 = air.wait_all async 
            %411 = air.wait_all async 
            %412 = air.wait_all async 
            %413 = air.wait_all async 
            %414 = air.wait_all async 
            %415 = air.wait_all async 
            %416 = air.wait_all async 
            %417 = air.wait_all async 
            %418 = air.wait_all async 
            %419 = air.wait_all async 
            %420 = air.wait_all async 
            %421 = air.wait_all async 
            %422 = air.wait_all async 
            %423 = air.wait_all async 
            %424 = air.wait_all async 
            %425 = air.wait_all async 
            %426 = air.wait_all async 
            %427 = air.wait_all async 
            %428 = air.wait_all async 
            %429 = air.wait_all async 
            %430 = air.wait_all async 
            %431 = air.wait_all async 
            %432 = air.wait_all async 
            %433 = air.wait_all async 
            %434 = air.wait_all async 
            %435 = air.wait_all async 
            %436 = air.wait_all async 
            %437 = air.wait_all async 
            %438 = air.wait_all async 
            %439 = air.wait_all async 
            %440 = air.wait_all async 
            %441 = air.wait_all async 
            %442 = air.wait_all async 
            %443 = air.wait_all async 
            %444 = air.wait_all async 
            %445 = air.wait_all async 
            %446 = air.wait_all async 
            %447 = air.wait_all async 
            %448 = air.wait_all async 
            %449 = air.wait_all async 
            %450 = air.wait_all async 
            %451 = air.wait_all async 
            %452 = air.wait_all async 
            %453 = air.wait_all async 
            %454 = air.wait_all async 
            %455 = air.wait_all async 
            %456 = air.wait_all async 
            %457 = air.wait_all async 
            %458 = air.wait_all async 
            %459 = air.wait_all async 
            %460 = air.wait_all async 
            %461 = air.wait_all async 
            %462 = air.wait_all async 
            %463 = air.wait_all async 
            %464 = air.wait_all async 
            %465 = air.wait_all async 
            %466 = air.wait_all async 
            %467 = air.wait_all async 
            %468 = air.wait_all async 
            %469 = air.wait_all async 
            %470 = air.wait_all async 
            %471 = air.wait_all async 
            %472 = air.wait_all async 
            %473 = air.wait_all async 
            %474 = air.wait_all async 
            %475 = air.wait_all async 
            %476 = air.wait_all async 
            %477 = air.wait_all async 
            %478 = air.wait_all async 
            %479 = air.wait_all async 
            %480 = air.wait_all async 
            %481 = air.wait_all async 
            %482 = air.wait_all async 
            %483 = air.wait_all async 
            %484 = air.wait_all async 
            %485 = air.wait_all async 
            %486 = air.wait_all async 
            %487 = air.wait_all async 
            %488 = air.wait_all async 
            %489 = air.wait_all async 
            %490 = air.wait_all async 
            %491 = air.wait_all async 
            %492 = air.wait_all async 
            %493 = air.wait_all async 
            %494 = air.wait_all async 
            %495 = air.wait_all async 
            %496 = air.wait_all async 
            %497 = air.wait_all async 
            %498 = air.wait_all async 
            %499 = air.wait_all async 
            %500 = air.wait_all async 
            %501 = air.wait_all async 
            %502 = air.wait_all async 
            %503 = air.wait_all async 
            %504 = air.wait_all async 
            %505 = air.wait_all async 
            %506 = air.wait_all async 
            %507 = air.wait_all async 
            %508 = air.wait_all async 
            %509 = air.wait_all async 
            %510 = air.wait_all async 
            %511 = air.wait_all async 
            %512 = air.wait_all async 
            %513 = air.wait_all async 
            %514 = air.wait_all async 
            %515 = air.wait_all async 
            %516 = air.wait_all async 
            %517 = air.wait_all async 
            %518 = air.wait_all async 
            %519 = air.wait_all async 
            %520 = air.wait_all async 
            %521 = air.wait_all async 
            %522 = air.wait_all async 
            %523 = air.wait_all async 
            %524 = air.wait_all async 
            %525 = air.wait_all async 
            %526 = air.wait_all async 
            %527 = air.wait_all async 
            %528 = air.wait_all async 
            %529 = air.wait_all async 
            %530 = air.wait_all async 
            %531 = air.wait_all async 
            %532 = air.wait_all async 
            %533 = air.wait_all async 
            %534 = air.wait_all async 
            %535 = air.wait_all async 
            %536 = air.wait_all async 
            %537 = air.wait_all async 
            %538 = air.wait_all async 
            %539 = air.wait_all async 
            %540 = air.wait_all async 
            %541 = air.wait_all async 
            %542 = air.wait_all async 
            %543 = air.wait_all async 
            %544 = air.wait_all async 
            %545 = air.wait_all async 
            %546 = air.wait_all async 
            %547 = air.wait_all async 
            %548 = air.wait_all async 
            %549 = air.wait_all async 
            %550 = air.wait_all async 
            %551 = air.wait_all async 
            %552 = air.wait_all async 
            %553 = air.wait_all async 
            %554 = air.wait_all async 
            %555 = air.wait_all async 
            %556 = air.wait_all async 
            %557 = air.wait_all async 
            %558 = air.wait_all async 
            %559 = air.wait_all async 
            %560 = air.wait_all async 
            %561 = air.wait_all async 
            %562 = air.wait_all async 
            %563 = air.wait_all async 
            %564 = air.wait_all async 
            %565 = air.wait_all async 
            %566 = air.wait_all async 
            %567 = air.wait_all async 
            %568 = air.wait_all async 
            %569 = air.wait_all async 
            %570 = air.wait_all async 
            %571 = air.wait_all async 
            %572 = air.wait_all async 
            %573 = air.wait_all async 
            %574 = air.wait_all async 
            %575 = air.wait_all async 
            %576 = air.wait_all async 
            %577 = air.wait_all async 
            %578 = air.wait_all async 
            %579 = air.wait_all async 
            %580 = air.wait_all async 
            %581 = air.wait_all async 
            %582 = air.wait_all async 
            %583 = air.wait_all async 
            %584 = air.wait_all async 
            %585 = air.wait_all async 
            %586 = air.wait_all async 
            %587 = air.wait_all async 
            %588 = air.wait_all async 
            %589 = air.wait_all async 
            %590 = air.wait_all async 
            %591 = air.wait_all async 
            %592 = air.wait_all async 
            %593 = air.wait_all async 
            %594 = air.wait_all async 
            %595 = air.wait_all async 
            %596 = air.wait_all async 
            %597 = air.wait_all async 
            %598 = air.wait_all async 
            %599 = air.wait_all async 
            %600 = air.wait_all async 
            %601 = air.wait_all async 
            %602 = air.wait_all async 
            %603 = air.wait_all async 
            %604 = air.wait_all async 
            %605 = air.wait_all async 
            %606 = air.wait_all async 
            %607 = air.wait_all async 
            %608 = air.wait_all async 
            %609 = air.wait_all async 
            %610 = air.wait_all async 
            %611 = air.wait_all async 
            %612 = air.wait_all async 
            %613 = air.wait_all async 
            %614 = air.wait_all async 
            %615 = air.wait_all async 
            %616 = air.wait_all async 
            %617 = air.wait_all async 
            %618 = air.wait_all async 
            %619 = air.wait_all async 
            %620 = air.wait_all async 
            %621 = air.wait_all async 
            %622 = air.wait_all async 
            %623 = air.wait_all async 
            %624 = air.wait_all async 
            %625 = air.wait_all async 
            %626 = air.wait_all async 
            %627 = air.wait_all async 
            %628 = air.wait_all async 
            %629 = air.wait_all async 
            %630 = air.wait_all async 
            %631 = air.wait_all async 
            %632 = air.wait_all async 
            %633 = air.wait_all async 
            %634 = air.wait_all async 
            %635 = air.wait_all async 
            %636 = air.wait_all async 
            %637 = air.wait_all async 
            %638 = air.wait_all async 
            %639 = air.wait_all async 
            %640 = air.wait_all async 
            %641 = air.wait_all async 
            %642 = air.wait_all async 
            %643 = air.wait_all async 
            %644 = air.wait_all async 
            %645 = air.wait_all async 
            %646 = air.wait_all async 
            %647 = air.wait_all async 
            %648 = air.wait_all async 
            %649 = air.wait_all async 
            %650 = air.wait_all async 
            %651 = air.wait_all async 
            %652 = air.wait_all async 
            %653 = air.wait_all async 
            %654 = air.wait_all async 
            %655 = air.wait_all async 
            %656 = air.wait_all async 
            %657 = air.wait_all async 
            %658 = air.wait_all async 
            %659 = air.wait_all async 
            %660 = air.wait_all async 
            %661 = air.wait_all async 
            %662 = air.wait_all async 
            %663 = air.wait_all async 
            %664 = air.wait_all async 
            %665 = air.wait_all async 
            %666 = air.wait_all async 
            %667 = air.wait_all async 
            %668 = air.wait_all async 
            %669 = air.wait_all async 
            %670 = air.wait_all async 
            %671 = air.wait_all async 
            %672 = air.wait_all async 
            %673 = air.wait_all async 
            %674 = air.wait_all async 
            %675 = air.wait_all async 
            %676 = air.wait_all async 
            %677 = air.wait_all async 
            %678 = air.wait_all async 
            %679 = air.wait_all async 
            %680 = air.wait_all async 
            %681 = air.wait_all async 
            %682 = air.wait_all async 
            %683 = air.wait_all async 
            %684 = air.wait_all async 
            %685 = air.wait_all async 
            %686 = air.wait_all async 
            %687 = air.wait_all async 
            %688 = air.wait_all async 
            %689 = air.wait_all async 
            %690 = air.wait_all async 
            %691 = air.wait_all async 
            %692 = air.wait_all async 
            %693 = air.wait_all async 
            %694 = air.wait_all async 
            %695 = air.wait_all async 
            %696 = air.wait_all async 
            %697 = air.wait_all async 
            %698 = air.wait_all async 
            %699 = air.wait_all async 
            %700 = air.wait_all async 
            %701 = air.wait_all async 
            %702 = air.wait_all async 
            %703 = air.wait_all async 
            %704 = air.wait_all async 
            %705 = air.wait_all async 
            %706 = air.wait_all async 
            %707 = air.wait_all async 
            %708 = air.wait_all async 
            %709 = air.wait_all async 
            %710 = air.wait_all async 
            %711 = air.wait_all async 
            %712 = air.wait_all async 
            %713 = air.wait_all async 
            %714 = air.wait_all async 
            %715 = air.wait_all async 
            %716 = air.wait_all async 
            %717 = air.wait_all async 
            %718 = air.wait_all async 
            %719 = air.wait_all async 
            %720 = air.wait_all async 
            %721 = air.wait_all async 
            %722 = air.wait_all async 
            %723 = air.wait_all async 
            %724 = air.wait_all async 
            %725 = air.wait_all async 
            %726 = air.wait_all async 
            %727 = air.wait_all async 
            %728 = air.wait_all async 
            %729 = air.wait_all async 
            %730 = air.wait_all async 
            %731 = air.wait_all async 
            %732 = air.wait_all async 
            %733 = air.wait_all async 
            %734 = air.wait_all async 
            %735 = air.wait_all async 
            %736 = air.wait_all async 
            %737 = air.wait_all async 
            %738 = air.wait_all async 
            %739 = air.wait_all async 
            %740 = air.wait_all async 
            %741 = air.wait_all async 
            %742 = air.wait_all async 
            %743 = air.wait_all async 
            %744 = air.wait_all async 
            %745 = air.wait_all async 
            %746 = air.wait_all async 
            %747 = air.wait_all async 
            %748 = air.wait_all async 
            %749 = air.channel.get async  @channel_15[] (%arg9[] [] []) : (memref<32x32xi32>)
            %750 = air.wait_all async 
            %751 = air.wait_all async 
            %752 = air.wait_all async 
            %753 = air.wait_all async 
            %754 = air.segment @forward_dispatch_0_matmul_32x32x32_i32_0 async  args(%arg10=%arg7, %arg11=%arg8, %arg12=%arg9) : memref<32x32xi32>, memref<32x32xi32>, memref<32x32xi32> attributes {id = 2 : i32} {
              %c7 = arith.constant 7 : index
              %c6 = arith.constant 6 : index
              %c5 = arith.constant 5 : index
              %c3 = arith.constant 3 : index
              %c128 = arith.constant 128 : index
              %c2 = arith.constant 2 : index
              %c1_1 = arith.constant 1 : index
              %c32_2 = arith.constant 32 : index
              %c256_3 = arith.constant 256 : index
              %c0_4 = arith.constant 0 : index
              %c8_5 = arith.constant 8 : index
              %c4_6 = arith.constant 4 : index
              %async_token, %results = air.execute -> (memref<4x8x8x4xi32, 1 : i32>) {
                %alloc = memref.alloc() : memref<4x8x8x4xi32, 1 : i32>
                air.execute_terminator %alloc : memref<4x8x8x4xi32, 1 : i32>
              } {id = 1 : i32}
              %async_token_7, %results_8 = air.execute -> (memref<8x1x32x4xi32, 1 : i32>) {
                %alloc = memref.alloc() : memref<8x1x32x4xi32, 1 : i32>
                air.execute_terminator %alloc : memref<8x1x32x4xi32, 1 : i32>
              } {id = 2 : i32}
              %async_token_9, %results_10 = air.execute -> (memref<4x1x8x32xi32, 1 : i32>) {
                %alloc = memref.alloc() : memref<4x1x8x32xi32, 1 : i32>
                air.execute_terminator %alloc : memref<4x1x8x32xi32, 1 : i32>
              } {id = 3 : i32}
              %755 = air.channel.get async [%async_token_9, %async_token_9]  @channel_0[] (%results_10[] [] []) : (memref<4x1x8x32xi32, 1 : i32>)
              %756 = air.wait_all async 
              %757 = air.channel.get async [%async_token_7, %async_token_7]  @channel_1[] (%results_8[] [] []) : (memref<8x1x32x4xi32, 1 : i32>)
              %758 = air.wait_all async 
              %759 = air.wait_all async 
              %760 = air.wait_all async 
              %761 = air.wait_all async 
              %762 = air.wait_all async 
              %763 = air.channel.put async [%async_token_9, %755]  @channel_2[] (%results_10[%c0_4, %c0_4, %c0_4, %c0_4, %c0_4, %c0_4] [%c1_1, %c1_1, %c4_6, %c2, %c4_6, %c8_5] [%c256_3, %c256_3, %c8_5, %c128, %c32_2, %c1_1]) {broadcast_set = #set} : (memref<4x1x8x32xi32, 1 : i32>)
              %764 = air.wait_all async 
              %765 = air.wait_all async 
              %766 = air.wait_all async 
              %767 = air.wait_all async 
              %768 = air.wait_all async 
              %769 = air.wait_all async 
              %770 = air.wait_all async 
              %771 = air.wait_all async 
              %772 = air.wait_all async 
              %773 = air.wait_all async 
              %774 = air.wait_all async 
              %775 = air.wait_all async 
              %776 = air.wait_all async 
              %777 = air.wait_all async 
              %778 = air.wait_all async 
              %779 = air.channel.put async [%async_token_9, %755]  @channel_3[] (%results_10[%c1_1, %c0_4, %c0_4, %c0_4, %c0_4, %c0_4] [%c1_1, %c1_1, %c4_6, %c2, %c4_6, %c8_5] [%c256_3, %c256_3, %c8_5, %c128, %c32_2, %c1_1]) {broadcast_set = #set1} : (memref<4x1x8x32xi32, 1 : i32>)
              %780 = air.wait_all async 
              %781 = air.wait_all async 
              %782 = air.wait_all async 
              %783 = air.wait_all async 
              %784 = air.wait_all async 
              %785 = air.wait_all async 
              %786 = air.wait_all async 
              %787 = air.wait_all async 
              %788 = air.wait_all async 
              %789 = air.wait_all async 
              %790 = air.wait_all async 
              %791 = air.wait_all async 
              %792 = air.wait_all async 
              %793 = air.wait_all async 
              %794 = air.wait_all async 
              %795 = air.wait_all async 
              %796 = air.wait_all async 
              %797 = air.channel.put async [%async_token_9, %755]  @channel_4[] (%results_10[%c2, %c0_4, %c0_4, %c0_4, %c0_4, %c0_4] [%c1_1, %c1_1, %c4_6, %c2, %c4_6, %c8_5] [%c256_3, %c256_3, %c8_5, %c128, %c32_2, %c1_1]) {broadcast_set = #set2} : (memref<4x1x8x32xi32, 1 : i32>)
              %798 = air.wait_all async 
              %799 = air.wait_all async 
              %800 = air.wait_all async 
              %801 = air.wait_all async 
              %802 = air.wait_all async 
              %803 = air.wait_all async 
              %804 = air.wait_all async 
              %805 = air.wait_all async 
              %806 = air.wait_all async 
              %807 = air.wait_all async 
              %808 = air.wait_all async 
              %809 = air.wait_all async 
              %810 = air.wait_all async 
              %811 = air.wait_all async 
              %812 = air.wait_all async 
              %813 = air.wait_all async 
              %814 = air.wait_all async 
              %815 = air.wait_all async 
              %816 = air.wait_all async 
              %817 = air.wait_all async 
              %818 = air.channel.put async [%async_token_9, %755]  @channel_5[] (%results_10[%c3, %c0_4, %c0_4, %c0_4, %c0_4, %c0_4] [%c1_1, %c1_1, %c4_6, %c2, %c4_6, %c8_5] [%c256_3, %c256_3, %c8_5, %c128, %c32_2, %c1_1]) {broadcast_set = #set3} : (memref<4x1x8x32xi32, 1 : i32>)
              %819 = air.wait_all async 
              %820 = air.wait_all async 
              %821 = air.wait_all async 
              %822 = air.wait_all async 
              %823 = air.wait_all async 
              %824 = air.wait_all async 
              %825 = air.wait_all async 
              %826 = air.wait_all async 
              %827 = air.wait_all async 
              %828 = air.wait_all async 
              %829 = air.wait_all async 
              %830 = air.wait_all async 
              %831 = air.wait_all async 
              %832 = air.channel.put async [%async_token_7, %757]  @channel_6[] (%results_8[%c0_4, %c0_4, %c0_4, %c0_4, %c0_4, %c0_4] [%c1_1, %c1_1, %c1_1, %c4_6, %c8_5, %c4_6] [%c128, %c128, %c4_6, %c32_2, %c4_6, %c1_1]) {broadcast_set = #set4} : (memref<8x1x32x4xi32, 1 : i32>)
              %833 = air.wait_all async 
              %834 = air.wait_all async 
              %835 = air.wait_all async 
              %836 = air.wait_all async 
              %837 = air.wait_all async 
              %838 = air.wait_all async 
              %839 = air.wait_all async 
              %840 = air.wait_all async 
              %841 = air.wait_all async 
              %842 = air.wait_all async 
              %843 = air.wait_all async 
              %844 = air.wait_all async 
              %845 = air.wait_all async 
              %846 = air.wait_all async 
              %847 = air.wait_all async 
              %848 = air.channel.put async [%async_token_7, %757]  @channel_7[] (%results_8[%c1_1, %c0_4, %c0_4, %c0_4, %c0_4, %c0_4] [%c1_1, %c1_1, %c1_1, %c4_6, %c8_5, %c4_6] [%c128, %c128, %c4_6, %c32_2, %c4_6, %c1_1]) {broadcast_set = #set5} : (memref<8x1x32x4xi32, 1 : i32>)
              %849 = air.wait_all async 
              %850 = air.wait_all async 
              %851 = air.wait_all async 
              %852 = air.wait_all async 
              %853 = air.wait_all async 
              %854 = air.wait_all async 
              %855 = air.wait_all async 
              %856 = air.wait_all async 
              %857 = air.wait_all async 
              %858 = air.wait_all async 
              %859 = air.wait_all async 
              %860 = air.wait_all async 
              %861 = air.wait_all async 
              %862 = air.wait_all async 
              %863 = air.wait_all async 
              %864 = air.wait_all async 
              %865 = air.wait_all async 
              %866 = air.channel.put async [%async_token_7, %757]  @channel_8[] (%results_8[%c2, %c0_4, %c0_4, %c0_4, %c0_4, %c0_4] [%c1_1, %c1_1, %c1_1, %c4_6, %c8_5, %c4_6] [%c128, %c128, %c4_6, %c32_2, %c4_6, %c1_1]) {broadcast_set = #set6} : (memref<8x1x32x4xi32, 1 : i32>)
              %867 = air.wait_all async 
              %868 = air.wait_all async 
              %869 = air.wait_all async 
              %870 = air.wait_all async 
              %871 = air.wait_all async 
              %872 = air.wait_all async 
              %873 = air.wait_all async 
              %874 = air.wait_all async 
              %875 = air.wait_all async 
              %876 = air.wait_all async 
              %877 = air.wait_all async 
              %878 = air.wait_all async 
              %879 = air.wait_all async 
              %880 = air.wait_all async 
              %881 = air.wait_all async 
              %882 = air.wait_all async 
              %883 = air.wait_all async 
              %884 = air.wait_all async 
              %885 = air.wait_all async 
              %886 = air.channel.put async [%async_token_7, %757]  @channel_9[] (%results_8[%c3, %c0_4, %c0_4, %c0_4, %c0_4, %c0_4] [%c1_1, %c1_1, %c1_1, %c4_6, %c8_5, %c4_6] [%c128, %c128, %c4_6, %c32_2, %c4_6, %c1_1]) {broadcast_set = #set7} : (memref<8x1x32x4xi32, 1 : i32>)
              %887 = air.wait_all async 
              %888 = air.wait_all async 
              %889 = air.wait_all async 
              %890 = air.wait_all async 
              %891 = air.wait_all async 
              %892 = air.wait_all async 
              %893 = air.wait_all async 
              %894 = air.wait_all async 
              %895 = air.wait_all async 
              %896 = air.wait_all async 
              %897 = air.wait_all async 
              %898 = air.wait_all async 
              %899 = air.wait_all async 
              %900 = air.wait_all async 
              %901 = air.wait_all async 
              %902 = air.wait_all async 
              %903 = air.wait_all async 
              %904 = air.wait_all async 
              %905 = air.wait_all async 
              %906 = air.wait_all async 
              %907 = air.wait_all async 
              %908 = air.channel.put async [%async_token_7, %757]  @channel_10[] (%results_8[%c4_6, %c0_4, %c0_4, %c0_4, %c0_4, %c0_4] [%c1_1, %c1_1, %c1_1, %c4_6, %c8_5, %c4_6] [%c128, %c128, %c4_6, %c32_2, %c4_6, %c1_1]) {broadcast_set = #set8} : (memref<8x1x32x4xi32, 1 : i32>)
              %909 = air.wait_all async 
              %910 = air.wait_all async 
              %911 = air.wait_all async 
              %912 = air.wait_all async 
              %913 = air.wait_all async 
              %914 = air.wait_all async 
              %915 = air.wait_all async 
              %916 = air.wait_all async 
              %917 = air.wait_all async 
              %918 = air.wait_all async 
              %919 = air.wait_all async 
              %920 = air.wait_all async 
              %921 = air.wait_all async 
              %922 = air.wait_all async 
              %923 = air.wait_all async 
              %924 = air.wait_all async 
              %925 = air.wait_all async 
              %926 = air.wait_all async 
              %927 = air.wait_all async 
              %928 = air.wait_all async 
              %929 = air.wait_all async 
              %930 = air.wait_all async 
              %931 = air.wait_all async 
              %932 = air.channel.put async [%async_token_7, %757]  @channel_11[] (%results_8[%c5, %c0_4, %c0_4, %c0_4, %c0_4, %c0_4] [%c1_1, %c1_1, %c1_1, %c4_6, %c8_5, %c4_6] [%c128, %c128, %c4_6, %c32_2, %c4_6, %c1_1]) {broadcast_set = #set9} : (memref<8x1x32x4xi32, 1 : i32>)
              %933 = air.wait_all async 
              %934 = air.wait_all async 
              %935 = air.wait_all async 
              %936 = air.wait_all async 
              %937 = air.wait_all async 
              %938 = air.wait_all async 
              %939 = air.wait_all async 
              %940 = air.wait_all async 
              %941 = air.wait_all async 
              %942 = air.wait_all async 
              %943 = air.wait_all async 
              %944 = air.wait_all async 
              %945 = air.wait_all async 
              %946 = air.wait_all async 
              %947 = air.wait_all async 
              %948 = air.wait_all async 
              %949 = air.wait_all async 
              %950 = air.wait_all async 
              %951 = air.wait_all async 
              %952 = air.wait_all async 
              %953 = air.wait_all async 
              %954 = air.wait_all async 
              %955 = air.wait_all async 
              %956 = air.wait_all async 
              %957 = air.wait_all async 
              %958 = air.channel.put async [%async_token_7, %757]  @channel_12[] (%results_8[%c6, %c0_4, %c0_4, %c0_4, %c0_4, %c0_4] [%c1_1, %c1_1, %c1_1, %c4_6, %c8_5, %c4_6] [%c128, %c128, %c4_6, %c32_2, %c4_6, %c1_1]) {broadcast_set = #set10} : (memref<8x1x32x4xi32, 1 : i32>)
              %959 = air.wait_all async 
              %960 = air.wait_all async 
              %961 = air.wait_all async 
              %962 = air.wait_all async 
              %963 = air.wait_all async 
              %964 = air.wait_all async 
              %965 = air.wait_all async 
              %966 = air.wait_all async 
              %967 = air.wait_all async 
              %968 = air.wait_all async 
              %969 = air.wait_all async 
              %970 = air.wait_all async 
              %971 = air.wait_all async 
              %972 = air.wait_all async 
              %973 = air.wait_all async 
              %974 = air.wait_all async 
              %975 = air.wait_all async 
              %976 = air.wait_all async 
              %977 = air.wait_all async 
              %978 = air.wait_all async 
              %979 = air.wait_all async 
              %980 = air.wait_all async 
              %981 = air.wait_all async 
              %982 = air.wait_all async 
              %983 = air.wait_all async 
              %984 = air.wait_all async 
              %985 = air.wait_all async 
              %986 = air.wait_all async 
              %987 = air.channel.put async [%async_token_7, %757]  @channel_13[] (%results_8[%c7, %c0_4, %c0_4, %c0_4, %c0_4, %c0_4] [%c1_1, %c1_1, %c1_1, %c4_6, %c8_5, %c4_6] [%c128, %c128, %c4_6, %c32_2, %c4_6, %c1_1]) {broadcast_set = #set11} : (memref<8x1x32x4xi32, 1 : i32>)
              %988 = air.wait_all async 
              %989 = air.wait_all async 
              %990 = air.wait_all async 
              %991 = air.wait_all async 
              %992 = air.wait_all async 
              %993 = air.wait_all async 
              %994 = air.wait_all async 
              %995 = air.wait_all async [%async_token, %async_token, %755, %757] 
              %996 = scf.parallel (%arg13, %arg14) = (%c0_4, %c0_4) to (%c4_6, %c8_5) step (%c1_1, %c1_1) init (%995) -> !air.async.token {
                %1000 = air.wait_all async 
                %1001 = air.wait_all async 
                %1002 = air.wait_all async 
                %1003 = air.wait_all async 
                %1004 = air.wait_all async 
                %1005 = air.wait_all async 
                %1006 = air.wait_all async 
                %1007 = air.channel.get async [%async_token, %995]  @channel_14[%arg13, %arg14] (%results[%arg13, %arg14, %c0_4, %c0_4] [%c1_1, %c1_1, %c8_5, %c4_6] [%c256_3, %c32_2, %c4_6, %c1_1]) : (memref<4x8x8x4xi32, 1 : i32>)
                %1008 = air.wait_all async 
                %1009 = air.wait_all async 
                %1010 = air.wait_all async 
                %1011 = air.wait_all async 
                %1012 = air.wait_all async [%1007] 
                scf.reduce(%1012 : !air.async.token) {
                ^bb0(%arg15: !air.async.token, %arg16: !air.async.token):
                  %1013 = air.wait_all async [%arg15, %arg16] 
                  scf.reduce.return %1013 : !air.async.token
                }
              }
              %997 = air.herd @herd_0 async [%async_token, %755, %757]  tile (%arg13, %arg14) in (%arg15=%c4_6, %arg16=%c8_5) args(%arg17=%results_10, %arg18=%results_8, %arg19=%results) : memref<4x1x8x32xi32, 1 : i32>, memref<8x1x32x4xi32, 1 : i32>, memref<4x8x8x4xi32, 1 : i32> attributes {id = 1 : i32} {
                %c16 = arith.constant 16 : index
                %c2_14 = arith.constant 2 : index
                %c4_15 = arith.constant 4 : index
                %c1_16 = arith.constant 1 : index
                %c32_17 = arith.constant 32 : index
                %c256_18 = arith.constant 256 : index
                %c0_19 = arith.constant 0 : index
                %cst = arith.constant dense<0> : vector<1x1x1x2x4x4xi32>
                %async_token_20, %results_21 = air.execute -> (memref<4x8x1x2x4x4xi32, 2 : i32>) {
                  %alloc = memref.alloc() : memref<4x8x1x2x4x4xi32, 2 : i32>
                  air.execute_terminator %alloc : memref<4x8x1x2x4x4xi32, 2 : i32>
                } {id = 4 : i32}
                %async_token_22, %results_23 = air.execute -> (memref<1x1x1x4x8x4xi32, 2 : i32>) {
                  %alloc = memref.alloc() : memref<1x1x1x4x8x4xi32, 2 : i32>
                  air.execute_terminator %alloc : memref<1x1x1x4x8x4xi32, 2 : i32>
                } {id = 5 : i32}
                %async_token_24, %results_25 = air.execute -> (memref<1x1x4x2x4x8xi32, 2 : i32>) {
                  %alloc = memref.alloc() : memref<1x1x4x2x4x8xi32, 2 : i32>
                  air.execute_terminator %alloc : memref<1x1x4x2x4x8xi32, 2 : i32>
                } {id = 6 : i32}
                %1000 = affine.if #set()[%arg13, %arg14] -> !air.async.token {
                  %1004 = air.channel.get async [%async_token_24, %async_token_24]  @channel_2[%arg13, %arg14] (%results_25[] [] []) : (memref<1x1x4x2x4x8xi32, 2 : i32>)
                  %1005 = air.wait_all async 
                  affine.yield %1004 : !air.async.token
                } else {
                  %1004 = affine.if #set1()[%arg13, %arg14] -> !air.async.token {
                    %1005 = air.channel.get async [%async_token_24, %async_token_24]  @channel_3[%arg13, %arg14] (%results_25[] [] []) : (memref<1x1x4x2x4x8xi32, 2 : i32>)
                    %1006 = air.wait_all async 
                    affine.yield %1005 : !air.async.token
                  } else {
                    %1005 = affine.if #set2()[%arg13, %arg14] -> !air.async.token {
                      %1006 = air.channel.get async [%async_token_24, %async_token_24]  @channel_4[%arg13, %arg14] (%results_25[] [] []) : (memref<1x1x4x2x4x8xi32, 2 : i32>)
                      %1007 = air.wait_all async 
                      affine.yield %1006 : !air.async.token
                    } else {
                      %1006 = air.channel.get async [%async_token_24, %async_token_24]  @channel_5[%arg13, %arg14] (%results_25[] [] []) : (memref<1x1x4x2x4x8xi32, 2 : i32>)
                      %1007 = air.wait_all async 
                      affine.yield %1006 : !air.async.token
                    }
                    affine.yield %1005 : !air.async.token
                  }
                  affine.yield %1004 : !air.async.token
                }
                %1001 = affine.if #set4()[%arg13, %arg14] -> !air.async.token {
                  %1004 = air.channel.get async [%async_token_22, %async_token_22]  @channel_6[%arg13, %arg14] (%results_23[] [] []) : (memref<1x1x1x4x8x4xi32, 2 : i32>)
                  %1005 = air.wait_all async 
                  affine.yield %1004 : !air.async.token
                } else {
                  %1004 = affine.if #set5()[%arg13, %arg14] -> !air.async.token {
                    %1005 = air.channel.get async [%async_token_22, %async_token_22]  @channel_7[%arg13, %arg14] (%results_23[] [] []) : (memref<1x1x1x4x8x4xi32, 2 : i32>)
                    %1006 = air.wait_all async 
                    affine.yield %1005 : !air.async.token
                  } else {
                    %1005 = affine.if #set6()[%arg13, %arg14] -> !air.async.token {
                      %1006 = air.channel.get async [%async_token_22, %async_token_22]  @channel_8[%arg13, %arg14] (%results_23[] [] []) : (memref<1x1x1x4x8x4xi32, 2 : i32>)
                      %1007 = air.wait_all async 
                      affine.yield %1006 : !air.async.token
                    } else {
                      %1006 = affine.if #set7()[%arg13, %arg14] -> !air.async.token {
                        %1007 = air.channel.get async [%async_token_22, %async_token_22]  @channel_9[%arg13, %arg14] (%results_23[] [] []) : (memref<1x1x1x4x8x4xi32, 2 : i32>)
                        %1008 = air.wait_all async 
                        affine.yield %1007 : !air.async.token
                      } else {
                        %1007 = affine.if #set8()[%arg13, %arg14] -> !air.async.token {
                          %1008 = air.channel.get async [%async_token_22, %async_token_22]  @channel_10[%arg13, %arg14] (%results_23[] [] []) : (memref<1x1x1x4x8x4xi32, 2 : i32>)
                          %1009 = air.wait_all async 
                          affine.yield %1008 : !air.async.token
                        } else {
                          %1008 = affine.if #set9()[%arg13, %arg14] -> !air.async.token {
                            %1009 = air.channel.get async [%async_token_22, %async_token_22]  @channel_11[%arg13, %arg14] (%results_23[] [] []) : (memref<1x1x1x4x8x4xi32, 2 : i32>)
                            %1010 = air.wait_all async 
                            affine.yield %1009 : !air.async.token
                          } else {
                            %1009 = affine.if #set10()[%arg13, %arg14] -> !air.async.token {
                              %1010 = air.channel.get async [%async_token_22, %async_token_22]  @channel_12[%arg13, %arg14] (%results_23[] [] []) : (memref<1x1x1x4x8x4xi32, 2 : i32>)
                              %1011 = air.wait_all async 
                              affine.yield %1010 : !air.async.token
                            } else {
                              %1010 = air.channel.get async [%async_token_22, %async_token_22]  @channel_13[%arg13, %arg14] (%results_23[] [] []) : (memref<1x1x1x4x8x4xi32, 2 : i32>)
                              %1011 = air.wait_all async 
                              affine.yield %1010 : !air.async.token
                            }
                            affine.yield %1009 : !air.async.token
                          }
                          affine.yield %1008 : !air.async.token
                        }
                        affine.yield %1007 : !air.async.token
                      }
                      affine.yield %1006 : !air.async.token
                    }
                    affine.yield %1005 : !air.async.token
                  }
                  affine.yield %1004 : !air.async.token
                }
                %subview = memref.subview %results_21[%arg13, %arg14, 0, 0, 0, 0] [1, 1, 1, 2, 4, 4] [1, 1, 1, 1, 1, 1] : memref<4x8x1x2x4x4xi32, 2 : i32> to memref<1x1x1x2x4x4xi32, strided<[256, 32, 32, 16, 4, 1], offset: ?>, 2 : i32>
                %async_token_26 = air.execute [%async_token_20] {
                  vector.transfer_write %cst, %subview[%c0_19, %c0_19, %c0_19, %c0_19, %c0_19, %c0_19] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x2x4x4xi32>, memref<1x1x1x2x4x4xi32, strided<[256, 32, 32, 16, 4, 1], offset: ?>, 2 : i32>
                } {id = 7 : i32}
                %async_token_27 = air.execute [%async_token_26, %1001, %1000] {
                  linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%results_25, %results_23 : memref<1x1x4x2x4x8xi32, 2 : i32>, memref<1x1x1x4x8x4xi32, 2 : i32>) outs(%subview : memref<1x1x1x2x4x4xi32, strided<[256, 32, 32, 16, 4, 1], offset: ?>, 2 : i32>) attrs =  {lowering_config = #config, packing_config = #packingConfig} {
                  ^bb0(%in: i32, %in_31: i32, %out: i32):
                    %1004 = arith.muli %in, %in_31 : i32
                    %1005 = arith.addi %out, %1004 : i32
                    linalg.yield %1005 : i32
                  }
                } {id = 8 : i32}
                %1002 = air.wait_all async 
                %1003 = air.channel.put async [%async_token_20, %async_token_20]  @channel_14[%arg13, %arg14] (%results_21[%arg13, %arg14, %c0_19, %c0_19, %c0_19, %c0_19] [%c1_16, %c1_16, %c2_14, %c4_15, %c1_16, %c4_15] [%c256_18, %c32_17, %c16, %c4_15, %c32_17, %c1_16]) : (memref<4x8x1x2x4x4xi32, 2 : i32>)
                %async_token_28 = air.execute [%async_token_27] {
                  memref.dealloc %results_25 : memref<1x1x4x2x4x8xi32, 2 : i32>
                } {id = 9 : i32}
                %async_token_29 = air.execute [%async_token_27] {
                  memref.dealloc %results_23 : memref<1x1x1x4x8x4xi32, 2 : i32>
                } {id = 10 : i32}
                %async_token_30 = air.execute [%1003] {
                  memref.dealloc %results_21 : memref<4x8x1x2x4x4xi32, 2 : i32>
                } {id = 11 : i32}
              }
              %998 = air.wait_all async 
              %999 = air.channel.put async [%async_token, %996, %997]  @channel_15[] (%results[%c0_4, %c0_4, %c0_4, %c0_4] [%c4_6, %c8_5, %c8_5, %c4_6] [%c256_3, %c4_6, %c32_2, %c1_1]) : (memref<4x8x8x4xi32, 1 : i32>)
              %async_token_11 = air.execute [%996] {
                memref.dealloc %results_10 : memref<4x1x8x32xi32, 1 : i32>
              } {id = 12 : i32}
              %async_token_12 = air.execute [%996] {
                memref.dealloc %results_8 : memref<8x1x32x4xi32, 1 : i32>
              } {id = 13 : i32}
              %async_token_13 = air.execute [%999] {
                memref.dealloc %results : memref<4x8x8x4xi32, 1 : i32>
              } {id = 14 : i32}
            }
          }
          return
        }
      }
    }
  }
  util.func public @forward(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @forward(%input0: tensor<32x32xi32>, %input1: tensor<32x32xi32>) -> (%output0: tensor<32x32xi32>)"}} {
    %c0 = arith.constant 0 : index
    %c4096 = arith.constant 4096 : index
    %c32 = arith.constant 32 : index
    %element_type_i32 = hal.element_type<i32> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%c32, %c32]) type(%element_type_i32) encoding(%dense_row_major)
    %0 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg0 : !hal.buffer_view -> tensor<32x32xi32> in !stream.resource<external>{%c4096}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input1") shape([%c32, %c32]) type(%element_type_i32) encoding(%dense_row_major)
    %1 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg1 : !hal.buffer_view -> tensor<32x32xi32> in !stream.resource<external>{%c4096}
    %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<external>{%c4096} => !stream.timepoint
    %2 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c4096}, %1 as %arg3: !stream.resource<external>{%c4096}, %result as %arg4: !stream.resource<external>{%c4096}) {
      stream.cmd.dispatch @forward_dispatch_0::@amdaie_pdi_fb::@forward_dispatch_0_matmul_32x32x32_i32 {
        ro %arg2[%c0 for %c4096] : !stream.resource<external>{%c4096},
        ro %arg3[%c0 for %c4096] : !stream.resource<external>{%c4096},
        wo %arg4[%c0 for %c4096] : !stream.resource<external>{%c4096}
      }
    } => !stream.timepoint
    %3 = stream.timepoint.await %2 => %result : !stream.resource<external>{%c4096}
    %4 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %3 : tensor<32x32xi32> in !stream.resource<external>{%c4096} -> !hal.buffer_view
    util.return %4 : !hal.buffer_view
  }
}


