(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)

(assert (and (and  ( =  in0 1)  ( =  in1 0))  ( =  in2 1)))

(check-sat)
(get-model)