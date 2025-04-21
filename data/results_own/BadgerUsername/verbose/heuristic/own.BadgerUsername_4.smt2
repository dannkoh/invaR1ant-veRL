(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)
(declare-const in3 Int)

(assert (and (and (and (and (and (and (and (not ( = in0 95))  ( =  in0 45)) (not ( = in1 95)))  ( =  in1 45)) (not ( = in2 95)))  ( =  in2 45)) (not ( = in3 95)))  ( =  in3 45)))

(check-sat)
(get-model)