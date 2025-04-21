(declare-const in0 Int)
(declare-const in1 Int)

(assert  ( <  in1 in0))

(check-sat)
(get-model)