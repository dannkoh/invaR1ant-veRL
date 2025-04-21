(declare-const in1x0 Int)
(declare-const in0x1 Int)

(assert  ( =  in1x0 in0x1))

(check-sat)
(get-model)