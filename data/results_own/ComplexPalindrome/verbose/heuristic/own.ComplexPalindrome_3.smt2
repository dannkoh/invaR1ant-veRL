(declare-const in0 Int)
(declare-const in2 Int)

(assert  ( <  in0 in2))

(check-sat)
(get-model)