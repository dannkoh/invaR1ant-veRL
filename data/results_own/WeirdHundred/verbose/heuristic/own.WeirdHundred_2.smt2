(declare-const in0 Int)
(declare-const in1 Int)

(assert (and (and (not ( = in0 1))  ( =  in0 0))  ( >=  in1 100)))

(check-sat)
(get-model)