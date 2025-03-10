(declare-const in6 Int)
(declare-const in5 Int)
(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)
(declare-const in4 Int)
(declare-const in3 Int)

(assert (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (not ( = in0 in1)) (not ( = in0 in2))) (not ( = in0 in3))) (not ( = in0 in4))) (not ( = in0 in5))) (not ( = in0 in6))) (not ( = in1 in2))) (not ( = in1 in3))) (not ( = in1 in4))) (not ( = in1 in5))) (not ( = in1 in6))) (not ( = in2 in3))) (not ( = in2 in4))) (not ( = in2 in5))) (not ( = in2 in6))) (not ( = in3 in4))) (not ( = in3 in5))) (not ( = in3 in6))) (not ( = in4 in5))) (not ( = in4 in6))) (not ( = in5 in6))))

(check-sat)
(get-model)