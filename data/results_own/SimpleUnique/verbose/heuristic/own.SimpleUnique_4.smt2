(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)
(declare-const in3 Int)

(assert (and (and (and (and (and (not ( = in0 in1)) (not ( = in0 in2))) (not ( = in0 in3))) (not ( = in1 in2))) (not ( = in1 in3))) (not ( = in2 in3))))

(check-sat)
(get-model)