(declare-const in0 Int)

(assert  ( <  in0 100))

(check-sat)
(get-model)