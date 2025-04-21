(declare-const in6 Int)
(declare-const in5 Int)
(declare-const in8 Int)
(declare-const in7 Int)
(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)
(declare-const in3 Int)

(assert (and (and (and  ( <  in0 in8)  ( <  in1 in7))  ( <  in2 in6))  ( <  in3 in5)))

(check-sat)
(get-model)