# Numerische Methoden der Physik Zusammenfassung

## Modelle

- **Stochastisches Modell**: definiert wie die "zufälligkeit" der Beobachtunen modeliert werden.
- **Funktionales Modell**: Modeliert die Abhängigkeiten der Messgrösen, kann geschrieben werden als $f(\underline{x}) = \underline{l}$

## Messfehler

- **Grobe Fehler:** Wenn die Differenz zwischen Messungen über der üblichen Toleranz liegt. können mit Detektionsferhahren gefilltert werden.
- **systematische Fehler:** Beeinflussen die Messung immer gleich. können behandelt werden durch:
  - kallibrieren
  - besonderer Messverfahren
  - modelierung des zusätzlichen parameters
- Zufällige Fehler: Beruht sich auf Unvollständigkeit des Messprozesses, können nur statistisch behandelt werden.
- **Ausreisser**: Grobe Fehler, oder unwahrscheinlicher grosser zufälliger Fehler.
- **Präzision:** Ausmass der übereinstummung von wiederhohlten Messungen
- **Richtigkeit:** Ausmass der übereinstimmung von Mittelwert der Messungen und referenzwert
- **Genauigkeit:** Ausmass der übereinstimmung von messungen zum referenzwert

![Präzision vs Richtigkeit vs Genauigkeit](<assets/Pasted image 20250221125238.png>)

## Bedeutung von $\sigma_0$

Da oft nur die **relativen Genauigkeiten** der Messungen bekannt sind – nicht jedoch der **absolute Standardfehler** –, definieren wir die Kovarianzmatrix der Beobachtungen als
$$
K_{ll} = \sigma_0^2 Q_{ll}
$$
Dabei ist $K_{ll}$ die Kovarianzmatrix, $Q_{ll}$ wird als **Kofaktorenmatrix** bezeichnet. Im Folgenden verwenden wir $Q_{ll}$ zur weiteren Berechnung.

## Die Gewichtsmatrix

Die Gewichtsmatrix soll lediglich die **relative Gewichtung** der Beobachtungen modellieren. Daher ist es sinnvoll, mit der Kofaktorenmatrix zu arbeiten – nicht mit der Kovarianzmatrix.
$$
P_{ll} = Q_{ll}^{-1}
$$
*Falls $Q_{ll}$ nicht regulär ist, verwendet man stattdessen die Pseudoinverse.*

## Die Designmatrix

Um ein lineares funktionales Modell zu formulieren, definiert man die **Designmatrix** $A$, welche die Parameter des Modells mit den Beobachtungen verknüpft. Es gilt:
$$
l - v = A x
$$
Dabei sind

- $l$ die Beobachtungen,
- $x$ die gesuchten Modellparameter,
- $v$ die **Residuen**, also die Abweichungen zwischen Modell und Messung. Observed $-$ Computed

## Die Normalengleichung

Ziel ist es, die **gewichtete Summe der Residuenquadrate** zu minimieren:
$$
\hat{x} = \arg \min_x v^T P v
$$
Setzt man $v = A x - l$ ein, ergibt sich:
$$
\hat{x} = \arg \min_x (A x - l)^T P (A x - l)
$$
Durch Ableiten nach $x$ und Nullsetzen des Gradienten erhält man die **Normalengleichung**:
$$
\hat{x} = (A^T P A)^{-1} A^T P l
$$

Dabei bezeichnen wir:

- **Normalgleichungsmatrix**:
  $$
  N = A^T P A
  $$
- **Rechte Seite des Gleichungssystems**:
  $$
  b = A^T P l
  $$

Somit lässt sich die Lösung auch schreiben als
$$
\hat{x} = N^{-1} b
$$

## Die Kofaktoren der Unbekannten

Aus dem linearen Fehlerfortpflanzungsgesetz ergibt sich, dass die Kofaktorenmatrix der Unbekannten $x$ gegeben ist durch:
$$
Q_{xx} = N^{-1}
$$

## Die Kovarianzmatrix der Unbekannten

Ist $\sigma_0$ bekannt, ergibt sich die Kovarianzmatrix der geschätzten Parameter als:
$$
K_{xx} = \sigma_0^2 Q_{xx}
$$

## Die Bedeutung des mittleren Fehlers der Gewichtseinheit a posteriori $m_0$

Falls $\sigma_0$ nicht bekannt ist, kann er **a posteriori** geschätzt werden. Diese Schätzung nennen wir $m_0$, sie ist definiert durch:
$$
m_0^2 = \frac{v^T P v}{n - u}
$$
wobei:

- $n$ die Anzahl der Beobachtungen $l$ ist,
- $u$ die Anzahl der geschätzten Parameter $x$.

## Vergleich mit $\sigma_0$

Falls ein **a priori**-Wert für $\sigma_0$ bekannt ist, kann man ihn mit $m_0$ vergleichen. Der Vergleichsausdruck ist bis auf den Faktor $n - u$ Chi-quadrat $\chi^2(n - u)$-verteilt:
$$
\frac{m_0^2}{\sigma_0^2}  \sim \frac{\chi^2(n - u)}{n-u}
$$

Daraus ergibt sich ein **Modelltest**: Für ein gegebenes Irrtumsrisiko $\alpha$ berechnet man den kritischen Wert $x_{1-\alpha}$, der die Gleichung erfüllt:
$$
\alpha = P(\chi^2(n - u) \leq x_{1-\alpha})
$$

Die Modellannahme ist akzeptiert, falls:
$$
\frac{m_0^2}{\sigma_0^2} < \frac{x_{1-\alpha}}{n - u}
\quad \Leftrightarrow \quad
v^T P v < \sigma_0^2 \, x_{1-\alpha}
$$

In python kann man diese wie folgt berechnen

```python
from scipy.stats.distributions import chi2
chi2.ppf(0.975, df=2)
```

## Nicht linearer Ausgleich

Man hat nun ein Funktionales Modell welches nicht linear von den **Parametern** $a_1, \dots a_u$ abhängt.

**Bsp:**

$$
h_j = a_0^2 + \frac{1}{a_1}t_1^2
$$

**Frage:** Wie sieht nun die Design Matrix aus?

Man linearisiert das System! (Taylorentwicklung in den Parametern)

$$
h(a_0, a_1) = h(\tilde{a}_0, \tilde{a}_1) + \left.\frac{\partial h}{\partial a_0}\right|_{(\tilde{a}_0, \tilde{a}_1)}\Delta a_0 + \left.\frac{\partial h}{\partial a_1}\right|_{(\tilde{a}_0, \tilde{a}_1)}\Delta a_1 + \mathcal{O}(\Delta a^2)
$$
So erhält man
$$
\Delta l = h_i-h(\tilde{a}_0, \tilde{a}_1) = A \Delta x
$$
man kann also nun eine Normalgleichung aufstellen der Form:
$$
\Delta x = N^{-1} b = (A^T P A)^{-1} A^T P \Delta l
$$

So kann man iterativ die Parameter $x$ bestimmen!

## Filtern

### Polynome filter

### Diskrete Fourier Transformation

Sei:
$$
a(t) = a_0 + \sum_{i=1}^{i=m} a_i \cos(i \omega t) + b_i \sin(i \omega t)
$$
mit
$$
\omega = \frac{2\pi}{t_N-t_1}
$$
Die Designmatrix ist nun Trivial aufzustellen.
