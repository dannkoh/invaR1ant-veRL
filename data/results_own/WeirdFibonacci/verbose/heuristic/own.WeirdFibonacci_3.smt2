(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)

(assert  ( =  in2 ( +  in0 in1)))

(check-sat)
(get-model)