(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)

(assert (and  ( =  in0 in1)  ( <  in1 in2)))

(check-sat)
(get-model)