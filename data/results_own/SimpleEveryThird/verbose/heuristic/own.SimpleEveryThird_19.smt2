(declare-const in6 Int)
(declare-const in9 Int)
(declare-const in12 Int)
(declare-const in15 Int)
(declare-const in18 Int)
(declare-const in3 Int)

(assert (and (and (and (and (and  ( =  in3 0)  ( =  in6 0))  ( =  in9 0))  ( =  in12 0))  ( =  in15 0))  ( =  in18 0)))

(check-sat)
(get-model)