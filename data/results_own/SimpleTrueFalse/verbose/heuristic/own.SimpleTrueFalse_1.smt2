(declare-const in0 Int)

(assert  ( =  in0 1))

(check-sat)
(get-model)