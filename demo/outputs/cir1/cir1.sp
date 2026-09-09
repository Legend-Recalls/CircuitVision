* CircuitVision cir1 (topology + assigned values)
* 0=GND(electrical, raw GND nets collapsed)
R0 N13 N31 1k  ; resistor
R1 N30 N32 1k  ; resistor
R2 N12 N30 1k  ; resistor
R3 N32 N31 1k  ; resistor
R4 N14 N31 1k  ; resistor
R5 N12 N13 1k  ; resistor
R6 N13 N30 1k  ; resistor
R7 N13 N14 1k  ; resistor
R8 N13 N32 1k  ; resistor
R9 N10 N12 1k  ; resistor
* --- models ---
.model DMOD D
.model QMODN NPN
.model QMODP PNP
.model MMODN NMOS
.model MMODP PMOS
.model SMOD SW()
.op
.end
