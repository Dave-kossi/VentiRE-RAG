# VentiRE — Feuille de route technique
### De prototype étudiant à projet présentable pour un stage Data Science / IA

**Contexte du projet** : Assistant RAG local en français, analysant la réglementation française de la ventilation (RE2020, arrêté du 24 mars 1982). Objectif : prototype à présenter en entretien de stage (Data Science / IA), avec une exigence forte de **robustesse technique générale**.

---

## 1. Diagnostic de l'état actuel

| Fichier | Rôle | Problèmes identifiés |
|---|---|---|
| `path_1` (parsing) | PDF → Markdown via LlamaParse | Un seul fichier en dur, pas de métadonnées de page conservées, API LlamaParse partiellement obsolète |
| `path_2` (indexation) | Markdown → Index vectoriel | Pas de `chunk_size`/`overlap` explicite, pas de métadonnées enrichies, embedding anglais sur corpus français |
| `path_4` (RAG query) | Retrieval hybride + reranking + LLM | Rechargement d'index fragile, pas de gestion d'erreur, sources jamais exposées |
| `app.py` (Streamlit) | Interface utilisateur | Pas de cache (rechargement à chaque question), pas d'affichage des sources, erreurs brutes affichées |

**Diagnostic central** : la chaîne de traçabilité de la source est cassée dès le parsing et jamais réparée en aval. C'est le point unique qui, une fois corrigé, améliore à la fois la fiabilité technique ET la qualité de la démo.

---

## 2. Démarches et améliorations par fichier

### 2.1 `path_1` — Parsing PDF → Markdown

**Améliorations à apporter :**

1. **Conserver les métadonnées de page/section**
   - *Aujourd'hui* : `f.write(doc.text)` jette toute métadonnée fournie par LlamaParse.
   - *À faire* : sérialiser `doc.metadata` (numéro de page notamment) en parallèle du texte, ou utiliser une sortie structurée (JSON) plutôt que du Markdown brut concaténé.
   - *Justification* : sans cette info, impossible d'afficher "Source : Arrêté 1982, page 3" à l'utilisateur final. C'est la base de toute crédibilité d'un système RAG réglementaire — un ingénieur ne fera jamais confiance à une réponse non vérifiable.

2. **Traiter tous les PDF d'un dossier, pas un fichier en dur**
   - *Aujourd'hui* : `PDF_INPUT` est un chemin unique codé en dur.
   - *À faire* : boucler sur `Path(PDF_DIR).glob("*.pdf")`.
   - *Justification* : avec 6 documents à terme, relancer un script à la main par fichier n'est plus tenable ni démontrable en entretien comme un pipeline "automatisé".

3. **Migrer vers la configuration LlamaParse à jour**
   - *Constat* : LlamaIndex a publié une API v2 structurée (`input_options`, `output_options`, `processing_options`) recommandée pour tous les nouveaux usages ; l'ancienne interface à plat (`LlamaParse(result_type=..., language=...)`) reste fonctionnelle mais n'est plus la voie principale documentée.
   - *À faire* : a minima, documenter explicitement quelle version d'API/SDK est utilisée dans le code (commentaire + `requirements.txt` pinné), pour éviter qu'une mise à jour silencieuse de la librairie casse le pipeline.
   - *Justification* : montre une vigilance sur la dette technique liée aux dépendances externes — point que les recruteurs techniques testent souvent ("que se passe-t-il si la lib change ?").

4. **Ajouter un mode "table-aware" explicite**
   - *À faire* : activer/vérifier l'extraction structurée des tableaux (grilles de débits réglementaires), qui est justement l'argument de vente de LlamaParse vs un simple extracteur PDF.
   - *Justification* : les tableaux de débits (cuisine, salle de bain, etc.) sont le cœur informationnel des arrêtés — une erreur de parsing ici corrompt silencieusement toutes les réponses sur ce sujet.

---

### 2.2 `path_2` — Indexation

**Améliorations à apporter :**

1. **Changer le modèle d'embedding : `bge-large-en-v1.5` → `BGE-M3`**
   - *Constat* : le modèle actuel est anglais ; le corpus est 100% français.
   - *Justification* : c'est probablement le bug le plus impactant et le plus invisible du pipeline actuel — un mismatch linguistique embedding/corpus dégrade la qualité du retrieval dense sans erreur visible. À présenter en entretien comme : *"j'ai identifié et corrigé un bug de cohérence linguistique qui dégradait silencieusement le retrieval."*

2. **Ajouter des métadonnées de classification au moment de l'indexation**
   - *À faire* : enrichir chaque document avec `doc_type` (ex. `reglementation`, `protocole`), `annee`, `segment` (`residentiel_neuf`, `residentiel_existant`, `ERP`).
   - *Justification* : avec l'extension du corpus à 6 documents touchant des segments différents (logement neuf, logement existant, tertiaire/ERP), le système doit pouvoir distinguer des sources qui peuvent sembler proches sémantiquement mais ne s'appliquent pas au même contexte réglementaire. Sans cette distinction, le système risque de citer un texte applicable au résidentiel pour une question sur un ERP.

3. **Définir `chunk_size` et `chunk_overlap` explicitement**
   - *Aujourd'hui* : dépendance totale à la segmentation naturelle du Markdown par en-têtes.
   - *À faire* : fixer des valeurs réfléchies, testées empiriquement (ex. 512-1024 tokens avec overlap de 50-100), en complément du découpage par en-têtes.
   - *Justification* : un texte réglementaire dense peut produire des chunks trop longs (perte de précision du retrieval) ou trop courts (perte de contexte) si on se fie uniquement à la structure Markdown brute.

4. **Logger le nombre de chunks par document source**
   - *À faire* : afficher un résumé (`X chunks générés pour le document Y`) en fin d'indexation.
   - *Justification* : permet de détecter immédiatement un échec de parsing silencieux (ex. un document qui ne génère que 2 chunks alors qu'il fait 50 pages signale un problème).

---

### 2.3 `path_4` — Moteur RAG (retrieval + reranking + LLM)

**Améliorations à apporter :**

1. **Remplacer le chargement d'index fragile**
   - *Aujourd'hui* : `VectorStoreIndex(nodes=[], storage_context=storage_context)`.
   - *À faire* : utiliser `load_index_from_storage(storage_context)`, la méthode standard documentée de LlamaIndex pour recharger un index persistant.
   - *Justification* : le pattern actuel fonctionne par effet de bord plutôt que par une API conçue pour ça — fragile face aux changements de version de la librairie, et signale en code review un manque de maîtrise des fondamentaux du framework.

2. **Ajouter une gestion d'erreurs explicite**
   - *À faire* : encadrer les appels à Ollama et Cohere avec des `try/except` dédiés, avec des messages clairs (`Ollama n'est pas démarré`, `Clé COHERE_API_KEY absente — reranking désactivé`).
   - *Justification* : en démo devant une entreprise, un service externe non disponible ne doit jamais produire une stack trace brute — ça détruit la crédibilité instantanément. Une gestion d'erreur propre est aussi un classique des entretiens techniques ("comment gères-tu les pannes de dépendances externes ?").

3. **Exposer les sources utilisées dans la réponse**
   - *À faire* : exploiter `response.source_nodes` (déjà fourni nativement par LlamaIndex) pour retourner, en plus du texte de réponse, la liste des documents/extraits/scores utilisés.
   - *Justification* : c'est le changement à plus fort ratio impact/effort de tout le projet. Transforme la perception de "chatbot qui invente peut-être" à "système vérifiable" — argument central pour tout usage réglementaire.

4. **Renforcer le prompt pour le contexte multi-documents**
   - *Aujourd'hui* : le prompt demande de citer la source "si l'information y est explicitement liée", sans garde-fou sur la cohérence contextuelle.
   - *À faire* : avec l'extension à 6 documents couvrant logement/ERP, neuf/existant, demander explicitement au LLM de vérifier que le contexte (type de bâtiment, segment réglementaire) correspond à la question avant de répondre, et de signaler une ambiguïté plutôt que de répondre avec un texte potentiellement inapplicable.
   - *Justification* : évite un risque réel et démontrable (mélanger des obligations résidentielles et ERP) qui apparaît seulement à partir d'un corpus multi-segments — bon argument pour montrer une compréhension fine des limites du RAG en contexte réglementaire.

5. **Réactiver `num_queries` du `QueryFusionRetriever` (à tester, pas systématique)**
   - *Aujourd'hui* : `num_queries=1` désactive la génération de requêtes alternatives.
   - *À faire* : tester `num_queries=2-3` sur le jeu d'évaluation (voir section 3) pour mesurer si la reformulation de requête améliore réellement le rappel, sans introduire de dérive.
   - *Justification* : décision a tester avec des chiffres, pas à trancher à l'intuition — exactement la posture "rigueur quantitative" qu'un profil Maths/Data Science doit démontrer.

---

### 2.4 `app.py` — Interface Streamlit

**Améliorations à apporter :**

1. **Mettre en cache le chargement du système RAG**
   - *À faire* : décorer l'initialisation (modèle d'embedding, index, retriever) avec `@st.cache_resource`.
   - *Justification* : sans cache, chaque question recharge tout le pipeline depuis disque — lenteur visible et reproductible en démo live, mauvaise impression immédiate.

2. **Afficher les sources sous chaque réponse**
   - *À faire* : ajouter un bloc dépliable (`st.expander`) listant les extraits sources et leur document d'origine, en s'appuyant sur le `response.source_nodes` exposé côté `path_4`.
   - *Justification* : rend visible aux yeux de l'entreprise cible la différence entre un simple wrapper LLM et un système RAG réellement traçable.

3. **Gérer les erreurs de façon différenciée**
   - *Aujourd'hui* : un seul bloc `except Exception as e` générique affiché tel quel à l'utilisateur.
   - *À faire* : distinguer "stockage absent" (déjà fait), "Ollama indisponible", "erreur de reranking", avec des messages adaptés à chaque cas.
   - *Justification* : une bonne gestion d'erreur différenciée est un signal de maturité d'ingénierie logicielle, indépendamment du sujet IA.

4. **Ajouter un indicateur de configuration active**
   - *À faire* : afficher en barre latérale si le reranking Cohere est actif, quel LLM est utilisé, et la taille du corpus chargé.
   - *Justification* : utile en démo pour répondre immédiatement à la question "et si Cohere n'est pas configuré, ça marche quand même ?" sans avoir à l'expliquer verbalement.

---

## 3. Apport complémentaire proposé : évaluation rigoureuse du RAG

**Ce qui manque aujourd'hui et qui est rarement fait par des projets RAG étudiants** : aucune mesure quantitative de la qualité du système.

**À construire :**
- Un jeu de 15-30 questions/réponses de référence sur le corpus, avec la source attendue connue à l'avance.
- Inclure volontairement des questions "pièges" exploitant l'ambiguïté inter-documents (ex. logement neuf vs existant, résidentiel vs ERP).
- Mesurer : la précision du retrieval (le bon document est-il dans le top-k ?), la fidélité de la réponse au contexte récupéré (faithfulness), et l'exactitude des sources citées.
- Outil recommandé : `ragas`, librairie standard d'évaluation RAG, ou une évaluation maison simple si on veut limiter les dépendances.

**Justification** : c'est l'argument qui différencie le plus un candidat Data Science. Dire "j'ai un RAG qui fonctionne" est commun ; dire "j'ai mesuré une précision de retrieval de X% et une fidélité de Y%, et voici comment le reranking améliore ces chiffres" démontre une rigueur quantitative directement transférable à un poste de Data Scientist ou d'ML Engineer.

---

## 4. Apport complémentaire proposé : rule engine minimal (compliance check)

**Constat** : un LLM ne doit pas être l'arbitre d'une conformité réglementaire chiffrée — il peut halluciner un seuil ou mal appliquer une règle conditionnelle.

**À construire (version simple, pas un moteur complet) :**
- Une fonction Python codant en dur 1 à 2 règles réelles extraites des textes (ex. débit minimal d'extraction pour une cuisine selon l'arrêté du 24 mars 1982, avec ses conditions d'application).
- Le LLM est utilisé pour **expliquer** le résultat du calcul (texte source, justification), pas pour le produire.

**Justification** : démontre la compréhension d'une limite fondamentale des LLM en contexte réglementaire/juridique — un calcul de conformité doit être déterministe et vérifiable, le LLM n'intervenant qu'en couche d'explication. C'est un point de discussion fort en entretien sur "où s'arrête la responsabilité d'un LLM dans un système métier".

---

## 5. Extension du corpus documentaire : 3 → 6 documents

**Documents à ajouter**, en restant strictement dans le thème ventilation/qualité de l'air (volontairement sans élargir vers RT2012, DPE, ou décret tertiaire, jugés hors-périmètre pour un projet de cette taille) :

1. **Arrêté du 24 août 2006** — aération des parties communes / logements existants (complète l'arrêté de 1982, qui couvre le logement neuf).
2. **Réglementation qualité de l'air intérieur en ERP** (arrêtés du 1er juin 2016 modifiés) — élargit le corpus du résidentiel vers le tertiaire.
3. **Arrêté du 29 juillet 2025** — sécurité incendie et ventilation en ERP, modifiant les prescriptions sur les réseaux de ventilation générale et la VMC double flux.

**Ce que ce passage à 6 documents change concrètement :**

| Aspect | Avec 3 documents | Avec 6 documents |
|---|---|---|
| Type de questions possibles | Mono-cas d'usage (logement neuf RE2020) | Questions nécessitant un arbitrage contextuel (neuf vs existant, résidentiel vs ERP) |
| Bénéfice du hybrid retrieval + reranking | Marginal (documents thématiquement proches) | Réellement démontrable et mesurable |
| Risque de confusion documentaire | Faible | Réel — justifie le metadata filtering |
| Qualité du jeu d'évaluation | Questions simples | Questions pièges inter-documents possibles |
| Narratif en entretien | "Chatbot Q&A sur un guide" | "Système qui arbitre entre sources réglementaires concurrentes selon le contexte" |

**Point de vigilance à mentionner en entretien** : la directive européenne 2024/1275 sur la performance énergétique des bâtiments, entrée en vigueur en 2025, doit être transposée en droit français en 2026 et renforcera les exigences de contrôle des systèmes de ventilation. Argument à formuler ainsi : *"mon architecture est pensée pour absorber un nouveau texte réglementaire sans refonte complète, ce qui anticipe la transposition à venir."*

---

## 6. Ce qui est volontairement écarté (et pourquoi)

| Proposition (issue d'une recommandation externe) | Pourquoi écartée à ce stade |
|---|---|
| Query Classifier + Query Router | Complexité non justifiée avec un corpus de 6 documents ; pertinent seulement à plus grande échelle |
| Enrichissement massif (RT2012, DPE, décret tertiaire, IoT, photovoltaïque) | Dilution du temps disponible ; mieux vaut un petit corpus mesuré qu'un grand corpus non évalué |
| Niveau 3 "Energy Decision Copilot" (capteurs IoT, BIM, monitoring temps réel) | Hors scope total pour un projet de stage solo ; risque de sonner comme une promesse marketing non défendable techniquement en entretien |
| Rule engine complet multi-règles | Chantier de plusieurs semaines ; une version simple (1-2 règles) suffit à démontrer le concept |

---

## 7. Plan d'exécution suggéré (3-4 semaines)

| Semaine | Contenu |
|---|---|
| 1 | Corrections de robustesse (`path_4`, `path_2`, `path_1`, `app.py`) + changement d'embedding vers BGE-M3 |
| 1-2 | Affichage des sources de bout en bout (parsing → indexation → query engine → interface) |
| 2-3 | Construction du jeu d'évaluation (15-30 questions) + mesure des métriques RAG |
| 3-4 | Ajout des 3 nouveaux documents + metadata filtering + rule engine minimal (1-2 règles) |

---

## 8. Ce que ce projet permet de raconter en entretien (synthèse)

- *"J'ai identifié et corrigé un bug silencieux d'incohérence linguistique embedding/corpus."*
- *"J'ai rendu mon système traçable : chaque réponse cite ses sources exactes, ce qui est indispensable en contexte réglementaire."*
- *"J'ai mesuré objectivement la qualité de mon retrieval et de mes réponses, pas juste vérifié empiriquement que ça marchait."*
- *"J'ai identifié une limite fondamentale des LLM (le calcul de conformité) et conçu une architecture hybride règles+LLM en conséquence."*
- *"J'ai construit un corpus volontairement restreint mais cohérent, en justifiant explicitement ce que j'ai choisi de ne pas couvrir."*
- *"J'ai anticipé l'évolution réglementaire européenne (directive EPBD 2024/1275) dans mes choix d'architecture."*

Ce sont des phrases qui démontrent une rigueur d'ingénierie ET une rigueur de raisonnement — exactement le double profil attendu d'un candidat Maths/Data Science en stage IA.
