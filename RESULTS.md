# LLTA-Bench Results

**Target**: ESP32-C6 @ 160MHz
**Generated**: Auto-generated from `report.json`

## Results Summary

| Category | Instructions | Latency | Throughput |
|----------|-------------|---------|------------|
| **Arithmetic** | ADD, ADDI, SUB, AND, ANDI, OR, ORI, XOR, XORI, SLT, SLTI, SLTIU, SLTU | 1 | 1 |
| **Shifts** | SLL, SLLI, SRA, SRAI, SRL, SRLI | 1 | 1 |
| **Compressed ALU** | C.ADD, C.ADDI, C.AND, C.ANDI, C.OR, C.XOR, C.SUB, C.MV, C.LI, C.SLLI, C.SRAI, C.SRLI | 1 | 1 |
| **Upper Immediate** | LUI, AUIPC | 1 | 1 |
| **Multiply (low)** | MUL | 1 | 1 |
| **Multiply (high)** | MULH, MULHSU, MULHU | 2 | 2 |
| **Division** | DIV, DIVU, REM, REMU | 10 | 10* |
| **Sign/Zero Extend** | SEXT.B, SEXT.H, ZEXT.H.RV32, ZEXT.H.RV64 | 2 | 2 |
| **Word Load** | LW, C.LW | 3 | 0-1 |
| **Atomic (AMO)** | AMOADD.W, AMOSWAP.W, etc. | 6 | 6 |
| **Branch (not-taken)** | BEQ, BNE | 1 | 1 |
| **Branch (taken/complex)** | BGE, BGEU, BLT, BLTU, C.BEQZ, C.BNEZ | 1-3 | 1-2 |
| **Jump (direct)** | C.J, C.JAL, JAL | 2-3 | 2 |

*Division is not pipelined, so throughput ≈ latency.

## Detailed Results

| Instruction | Assembly | Type | Latency | Throughput |
|-------------|----------|------|---------|------------|
| ADD | `add a0, a0, a0` | arithmetic | 1 | 1 |
| ADDI | `addi a0, a0, 0` | arithmetic | 1 | 1 |
| AMOADD_W | `amoadd.w a0, a0, (a0)` | atomic | 6 | 6 |
| AMOADD_W_AQ | `amoadd.w.aq a0, a0, (a0)` | atomic | 6 | 6 |
| AMOADD_W_AQ_RL | `amoadd.w.aqrl a0, a0, (a0)` | atomic | 6 | 6 |
| AMOADD_W_RL | `amoadd.w.rl a0, a0, (a0)` | atomic | 6 | 6 |
| AMOAND_W | `amoand.w a0, a0, (a0)` | atomic | 6 | 6 |
| AMOAND_W_AQ | `amoand.w.aq a0, a0, (a0)` | atomic | 6 | 6 |
| AMOAND_W_AQ_RL | `amoand.w.aqrl a0, a0, (a0)` | atomic | 6 | 6 |
| AMOAND_W_RL | `amoand.w.rl a0, a0, (a0)` | atomic | 6 | 6 |
| AMOMAXU_W | `amomaxu.w a0, a0, (a0)` | atomic | 6 | 6 |
| AMOMAXU_W_AQ | `amomaxu.w.aq a0, a0, (a0)` | atomic | 6 | 6 |
| AMOMAXU_W_AQ_RL | `amomaxu.w.aqrl a0, a0, (a0)` | atomic | 6 | 6 |
| AMOMAXU_W_RL | `amomaxu.w.rl a0, a0, (a0)` | atomic | 6 | 6 |
| AMOMAX_W | `amomax.w a0, a0, (a0)` | atomic | 6 | 6 |
| AMOMAX_W_AQ | `amomax.w.aq a0, a0, (a0)` | atomic | 6 | 6 |
| AMOMAX_W_AQ_RL | `amomax.w.aqrl a0, a0, (a0)` | atomic | 6 | 6 |
| AMOMAX_W_RL | `amomax.w.rl a0, a0, (a0)` | atomic | 6 | 6 |
| AMOMINU_W | `amominu.w a0, a0, (a0)` | atomic | 6 | 6 |
| AMOMINU_W_AQ | `amominu.w.aq a0, a0, (a0)` | atomic | 6 | 6 |
| AMOMINU_W_AQ_RL | `amominu.w.aqrl a0, a0, (a0)` | atomic | 6 | 6 |
| AMOMINU_W_RL | `amominu.w.rl a0, a0, (a0)` | atomic | 6 | 6 |
| AMOMIN_W | `amomin.w a0, a0, (a0)` | atomic | 6 | 6 |
| AMOMIN_W_AQ | `amomin.w.aq a0, a0, (a0)` | atomic | 6 | 6 |
| AMOMIN_W_AQ_RL | `amomin.w.aqrl a0, a0, (a0)` | atomic | 6 | 6 |
| AMOMIN_W_RL | `amomin.w.rl a0, a0, (a0)` | atomic | 6 | 6 |
| AMOOR_W | `amoor.w a0, a0, (a0)` | atomic | 6 | 6 |
| AMOOR_W_AQ | `amoor.w.aq a0, a0, (a0)` | atomic | 6 | 6 |
| AMOOR_W_AQ_RL | `amoor.w.aqrl a0, a0, (a0)` | atomic | 6 | 6 |
| AMOOR_W_RL | `amoor.w.rl a0, a0, (a0)` | atomic | 6 | 6 |
| AMOSWAP_W | `amoswap.w a0, a0, (a0)` | atomic | 6 | 6 |
| AMOSWAP_W_AQ | `amoswap.w.aq a0, a0, (a0)` | atomic | 6 | 6 |
| AMOSWAP_W_AQ_RL | `amoswap.w.aqrl a0, a0, (a0)` | atomic | 6 | 6 |
| AMOSWAP_W_RL | `amoswap.w.rl a0, a0, (a0)` | atomic | 6 | 6 |
| AMOXOR_W | `amoxor.w a0, a0, (a0)` | atomic | 6 | 6 |
| AMOXOR_W_AQ | `amoxor.w.aq a0, a0, (a0)` | atomic | 6 | 6 |
| AMOXOR_W_AQ_RL | `amoxor.w.aqrl a0, a0, (a0)` | atomic | 6 | 6 |
| AMOXOR_W_RL | `amoxor.w.rl a0, a0, (a0)` | atomic | 6 | 6 |
| AND | `and a0, a0, a0` | arithmetic | 1 | 1 |
| ANDI | `andi a0, a0, 0` | arithmetic | 1 | 1 |
| AUIPC | `auipc a0, 0` | unknown | 1 | 1 |
| BEQ | `beq a0, a0, 0` | branch | 1 | 1 |
| BGE | `bge a0, a0, 0` | branch | 1 | 2 |
| BGEU | `bgeu a0, a0, 0` | branch | 1 | 2 |
| BLT | `blt a0, a0, 0` | branch | 1 | 1 |
| BLTU | `bltu a0, a0, 0` | branch | 1 | 1 |
| BNE | `bne a0, a0, 0` | branch | 1 | 1 |
| BRANCH_BACKWARD_NOT_TAKEN | `Branch Predictor Test` | branch | 4 | - |
| BRANCH_BACKWARD_TAKEN | `Branch Predictor Test` | branch | 2 | - |
| BRANCH_FORWARD_NOT_TAKEN | `Branch Predictor Test` | branch | 1 | - |
| BRANCH_FORWARD_TAKEN | `Branch Predictor Test` | branch | 4 | - |
| BRANCH_WARMUP_DETECTION | `Branch Predictor Test` | branch | 5 | - |
| CACHE_FLASH_HIT | `Pointer Chasing` | load | 4 | - |
| CACHE_FLASH_MISS | `Pointer Chasing` | load | 347 | - |
| CACHE_REPLACEMENT_LRU | `Pointer Chasing` | load | 123 | - |
| CACHE_SRAM | `Pointer Chasing` | load | 3 | - |
| C_ADD | `c.add a0, a0` | arithmetic | 1 | 1 |
| C_ADDI | `c.addi a0, 1` | arithmetic | 1 | 1 |
| C_ADDI_HINT_IMM_ZERO | `c.addi a0, 0` | arithmetic | 1 | 1 |
| C_ADD_HINT | `c.add zero, a0` | arithmetic | - | 1 |
| C_AND | `c.and a0, a0` | arithmetic | 1 | 1 |
| C_ANDI | `c.andi a0, 0` | arithmetic | 1 | 1 |
| C_BEQZ | `c.beqz a0, 0` | branch | 1 | 2 |
| C_BNEZ | `c.bnez a0, 0` | branch | 3 | 1 |
| C_J | `c.j 0` | jump | 2 | 2 |
| C_JAL | `c.jal 0` | jump | 2 | 2 |
| C_LI | `c.li a0, 0` | arithmetic | 1 | 1 |
| C_LI_HINT | `c.li zero, 0` | arithmetic | - | 1 |
| C_LW | `c.lw a0, 0(a0)` | load_store | 3 | 1 |
| C_LWSP | `c.lwsp a0, 0(sp)` | load_store | - | 1 |
| C_LWSP_INX | `c.lwsp a0, 0(sp)` | load_store | - | 1 |
| C_LW_INX | `c.lw a0, 0(a0)` | load_store | 3 | 1 |
| C_MV | `c.mv a0, a0` | arithmetic | 1 | 1 |
| C_MV_HINT | `c.mv zero, a0` | arithmetic | - | 1 |
| C_OR | `c.or a0, a0` | arithmetic | 1 | 1 |
| C_SLLI | `c.slli a0, 1` | arithmetic | 1 | 1 |
| C_SLLI_HINT | `c.slli zero, 1` | arithmetic | - | 1 |
| C_SRAI | `c.srai a0, 1` | arithmetic | 1 | 1 |
| C_SRLI | `c.srli a0, 1` | arithmetic | 1 | 1 |
| C_SUB | `c.sub a0, a0` | arithmetic | 1 | 1 |
| C_SW | `c.sw a0, 0(a0)` | load_store | 3 | 1 |
| C_SWSP | `c.swsp a0, 0(sp)` | load_store | - | 1 |
| C_SWSP_INX | `c.swsp a0, 0(sp)` | load_store | - | 1 |
| C_SW_INX | `c.sw a0, 0(a0)` | load_store | 3 | 1 |
| C_XOR | `c.xor a0, a0` | arithmetic | 1 | 1 |
| DIV | `div a0, a0, a0` | multiply | 10 | 10 |
| DIVU | `divu a0, a0, a0` | multiply | 10 | 10 |
| FETCH_BRANCHY_1000_JMP | `Instruction Fetch` | load | 3001 | - |
| FETCH_LINEAR_1000_NOP | `Instruction Fetch` | load | 1001 | - |
| JAL | `jal a0, 0` | jump | 3 | 2 |
| LB | `lb a0, 0(a0)` | load | - | 0 |
| LBU | `lbu a0, 0(a0)` | load | - | 0 |
| LH | `lh a0, 0(a0)` | load | - | 0 |
| LHU | `lhu a0, 0(a0)` | load | - | 0 |
| LH_INX | `lh a0, 0(a0)` | load | - | 0 |
| LUI | `lui a0, 0` | unknown | 1 | 1 |
| LW | `lw a0, 0(a0)` | load | 3 | 0 |
| LW_INX | `lw a0, 0(a0)` | load | 3 | 0 |
| MUL | `mul a0, a0, a0` | multiply | 1 | 1 |
| MULH | `mulh a0, a0, a0` | multiply | 2 | 2 |
| MULHSU | `mulhsu a0, a0, a0` | multiply | 2 | 2 |
| MULHU | `mulhu a0, a0, a0` | multiply | 2 | 2 |
| OR | `or a0, a0, a0` | arithmetic | 1 | 1 |
| ORI | `ori a0, a0, 0` | arithmetic | 1 | 1 |
| REM | `rem a0, a0, a0` | multiply | 10 | 10 |
| REMU | `remu a0, a0, a0` | multiply | 10 | 10 |
| SB | `sb a0, 0(a0)` | store | 3 | 1 |
| SB_BURST_100_SW | `sw a0, 0(a1) (x100)` | store | - | 0 |
| SB_FORWARDING_SW_LW | `sw + lw (same addr)` | load_store | 0 | - |
| SEXT_B | `sext.b a0, a0` | unknown | 2 | 2 |
| SEXT_H | `sext.h a0, a0` | unknown | 2 | 2 |
| SH | `sh a0, 0(a0)` | store | 3 | 1 |
| SH_INX | `sh a0, 0(a0)` | store | 3 | 1 |
| SLL | `sll a0, a0, a0` | arithmetic | 1 | 1 |
| SLLI | `slli a0, a0, 1` | arithmetic | 1 | 1 |
| SLT | `slt a0, a0, a0` | arithmetic | 1 | 1 |
| SLTI | `slti a0, a0, 0` | arithmetic | 1 | 1 |
| SLTIU | `sltiu a0, a0, 0` | arithmetic | 1 | 1 |
| SLTU | `sltu a0, a0, a0` | arithmetic | 1 | 1 |
| SRA | `sra a0, a0, a0` | arithmetic | 1 | 1 |
| SRAI | `srai a0, a0, 1` | arithmetic | 1 | 1 |
| SRL | `srl a0, a0, a0` | arithmetic | 1 | 1 |
| SRLI | `srli a0, a0, 1` | arithmetic | 1 | 1 |
| STRUCTURAL_HAZARD_DIV_ADD | `div + add (independent)` | arithmetic | - | 11 |
| SUB | `sub a0, a0, a0` | arithmetic | 1 | 1 |
| SW | `sw a0, 0(a0)` | store | 3 | 1 |
| SW_INX | `sw a0, 0(a0)` | store | 3 | 1 |
| UNALIGNED_LOAD_0_ALIGNED | `Unaligned Load` | load | 1 | - |
| UNALIGNED_LOAD_1 | `Unaligned Load` | load | 2 | - |
| UNALIGNED_LOAD_31_CROSS_LINE | `Unaligned Load` | load | 2 | - |
| XOR | `xor a0, a0, a0` | arithmetic | 1 | 1 |
| XORI | `xori a0, a0, 0` | arithmetic | 1 | 1 |
| ZEXT_H_RV32 | `zext.h a0, a0` | unknown | 2 | 2 |
| ZEXT_H_RV64 | `zext.h a0, a0` | unknown | 2 | 2 |
