(declare-const in0 Int)
(declare-const in1 Int)

(assert  ( =  in0 in1))

(check-sat)
(get-model)