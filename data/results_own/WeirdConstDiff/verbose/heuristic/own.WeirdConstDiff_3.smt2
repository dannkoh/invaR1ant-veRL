(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)

(assert  ( =  ( -  in2 in1) ( -  in1 in0)))

(check-sat)
(get-model)