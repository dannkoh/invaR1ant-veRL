(declare-const in0 Int)
(declare-const in1 Int)

(assert (and  ( =  in0 120)  ( =  in1 120)))

(check-sat)
(get-model)