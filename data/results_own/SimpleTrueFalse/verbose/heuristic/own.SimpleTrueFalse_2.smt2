(declare-const in0 Int)
(declare-const in1 Int)

(assert (and  ( =  in0 1)  ( =  in1 0)))

(check-sat)
(get-model)