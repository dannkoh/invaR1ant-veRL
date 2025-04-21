(declare-const in0 Int)

(assert  ( =  in0 120))

(check-sat)
(get-model)