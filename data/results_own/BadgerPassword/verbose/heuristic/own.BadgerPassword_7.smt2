(declare-const in6 Int)
(declare-const in5 Int)
(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)
(declare-const in4 Int)
(declare-const in3 Int)

(assert (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (not ( = in0 64)) (not ( = in0 35))) (not ( = in0 36)))  ( =  in0 37)) (not ( = in1 64))) (not ( = in1 35))) (not ( = in1 36)))  ( =  in1 37)) (not ( = in2 64))) (not ( = in2 35))) (not ( = in2 36)))  ( =  in2 37)) (not ( = in3 64))) (not ( = in3 35))) (not ( = in3 36)))  ( =  in3 37)) (not ( = in4 64))) (not ( = in4 35))) (not ( = in4 36)))  ( =  in4 37)) (not ( = in5 64))) (not ( = in5 35))) (not ( = in5 36)))  ( =  in5 37)) (not ( = in6 64))) (not ( = in6 35))) (not ( = in6 36)))  ( =  in6 37)))

(check-sat)
(get-model)