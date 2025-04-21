(declare-const in0 Int)

(assert (and  ( >=  in0 97)  ( <=  in0 122)))

(check-sat)
(get-model)