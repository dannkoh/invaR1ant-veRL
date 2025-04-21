(declare-const in6 Int)
(declare-const in5 Int)
(declare-const in8 Int)
(declare-const in7 Int)
(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)
(declare-const in4 Int)
(declare-const in3 Int)

(assert (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (not ( = in0 in1)) (not ( = in0 in2))) (not ( = in0 in3))) (not ( = in0 in4))) (not ( = in0 in5))) (not ( = in0 in6))) (not ( = in0 in7))) (not ( = in0 in8))) (not ( = in1 in2))) (not ( = in1 in3))) (not ( = in1 in4))) (not ( = in1 in5))) (not ( = in1 in6))) (not ( = in1 in7))) (not ( = in1 in8))) (not ( = in2 in3))) (not ( = in2 in4))) (not ( = in2 in5))) (not ( = in2 in6))) (not ( = in2 in7))) (not ( = in2 in8))) (not ( = in3 in4))) (not ( = in3 in5))) (not ( = in3 in6))) (not ( = in3 in7))) (not ( = in3 in8))) (not ( = in4 in5))) (not ( = in4 in6))) (not ( = in4 in7))) (not ( = in4 in8))) (not ( = in5 in6))) (not ( = in5 in7))) (not ( = in5 in8))) (not ( = in6 in7))) (not ( = in6 in8))) (not ( = in7 in8))))

(check-sat)
(get-model)