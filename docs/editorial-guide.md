# Editorial guide

This journal has one author, and every text in it should sound like her: posts, lessons, portfolio pages,
descriptions, figure titles and covers. The editor's job is to keep that voice whole. That means removing what
sounds generated, templated or bureaucratic and leaving everything that is hers, rough edges included.

The editor works on finished drafts. The editor does not add ideas, does not change facts, and does not make the
text "nicer". The editor makes it sound like one specific person wrote it.

`bin/style-lint.py` finds the mechanical part (the words and constructions catalogued below). The rest needs a
reader.

## The author

**Who speaks.** A researcher who did the work herself: ran the experiment, read the papers, broke the model. In
posts she says "I" for her own choices and opinions ("when I took parts of the input away from its smaller
sibling"). In lessons she says "we" for the work done together in the notebook and "you" for the reader. One
reader, never "readers" or "the audience".

**Temperament.**

- Curious. She asks the question herself and then goes and checks.
- Precise. Names, dates, exact quotes, numbers with their source.
- Sceptical of hype, including her own field's.
- Dry irony, sometimes sarcasm, aimed at methods, hype and her own model's failures. Never at the reader.
- Warmth shows through specificity and care, not through exclamation marks.

**Habits to keep.**

- She opens with something concrete: a scene with a date ("On 19 May 1972, at a symposium..."), a person, a
  number, an output the model got wrong.
- She lets primary sources speak in short exact quotes. She names the people behind an idea.
- She states assumptions and limits plainly ("which makes the question in the title less rhetorical than it
  sounds").
- She ends a section on its last real point, not on a moral.

**Rhythm.** Short paragraphs of one to three sentences, with varied sentence length. A sentence may start with
"And" or "But". An aside in parentheses is fine. Not every sentence needs to be smooth: smoothing everything into
one even tone is what makes a text read as generated.

**Typography.** Emphasis inside a sentence is italics only. There is no bold, highlighter or coloured text inside
sentences. Bold is reserved for the journal's components: the **Key point.** and **Trap.** callouts, and a term
defined for the first time in a lesson. Use at most one em dash per paragraph; a comma, a colon or parentheses
usually do the job better.

## What the editor removes

### Words

Each of these words has an honest use. In this journal they almost always arrive as filler, so each occurrence
needs a reason to stay. The usual fix is to delete the word, or to replace it with the specific fact it was
standing in for.

| Generated | Write instead |
|---|---|
| delve, dive into, deep dive, unpack, explore (as a promise) | say what you will show, or just show it |
| crucial, vital, pivotal, paramount, key (as an adjective of praise) | say what depends on it |
| robust, seamless, powerful, cutting-edge, state-of-the-art, groundbreaking, revolutionary, game-changer, transformative | a number, a comparison, or nothing |
| landscape, realm, tapestry, journey, ecosystem, paradigm shift | the actual field, set or change |
| leverage, harness, unlock, unleash, empower, elevate, foster, bolster | use, allow, help, raise |
| intricate, nuanced, multifaceted, holistic | the specific complication |
| testament to, stands as, serves as a reminder, sheds light on, paves the way | what was shown |
| it's worth noting, notably, importantly, interestingly, essentially, fundamentally, ultimately, at its core | delete; if a point matters, put it first |
| a myriad of, a plethora of, a wide range of, various, numerous | the number, or the two examples that matter |
| in today's fast-paced world, in the era of AI, ever-evolving | delete |

### Constructions

These patterns give generated text away faster than any single word.

1. **The contrast reveal.** "It's not just X, it's Y." "This isn't magic. It's math." "X? No. Y." Say Y.
2. **The rhetorical question answered in the next breath.** "The result? A 40% drop." "Why does this matter?
   Because..." State the result. A real question that the text leaves open for a while is fine.
3. **The aphorism that closes a paragraph.** "And that changes everything." "So the dress is the photo." The
   paragraph ends on its last fact.
4. **Triads by reflex.** "Fast, cheap and reliable." "Clear, concise and compelling." Keep the one adjective you
   can defend.
5. **Signposting and recaps.** "In this post we will...", "Let's take a look", "Let's break it down", "Here's the
   thing", "In conclusion", "To sum up", "Key takeaways". The structure should be visible without announcements.
6. **Staged suspense.** "Spoiler:", "Plot twist:", "Enter X.", "This is where X comes in.", "Buckle up." Remove
   them and let the fact do the work.
7. **Hedging stacks and empty attribution.** "may potentially", "could arguably", "studies show", "experts agree".
   Name the study or remove the claim.
8. **Audience symmetry.** "Whether you're a beginner or an expert..." Write for the one reader the text is for.
9. **Imagine-openings.** "Imagine a world where..." Open with the real scene instead.
10. **Anthropomorphic overclaim.** "The model wants / understands / believes", unless the sentence is about
    exactly that question. Say what the model computes or outputs.
11. **Two short sentences in contrast as a pair**, used as a rhythm device in every section.

### Structure

- **No bullet or numbered lists in the body of posts.** Write short prose paragraphs. Exceptions: the reference
  list, the step list at the top of a lesson ("What you'll do"), and genuinely sequential procedures (commands to
  run).
- **Titles and headings are one plain clause** containing the words a reader would search for. Never use
  "Topic: tagline" colon titles, "X, explained / measured / revisited", clickbait questions ("Is BLEU lying?") or
  slogans ("One photo, three products"). A plain question is fine when it is the real question of the text. The
  same rules apply to H2 and H3 headings, figure titles, descriptions and cover text.
- **No heading for every two paragraphs.** A section earns a heading when it has its own question.
- **No "Conclusion" that repeats the text.** If the end needs a section, it says something new: what remains
  open, or what the author would try next.
- **Never mention earlier versions, drafts or their errors.** Write the correct thing.
- Paragraphs longer than about five sentences get split. A long section without a figure is a warning sign.

### Russian texts

The same rules apply, plus two habits peculiar to Russian. One is bureaucratic style, *канцелярит*, as Nora Gal
described it in *Слово живое и мёртвое*. The other is the Russian translation of the generated patterns above.

| Generated or bureaucratic | Write instead |
|---|---|
| является, представляет собой | a dash, or a verb |
| данный, вышеуказанный | этот |
| осуществлять, производить (действие), в рамках, на сегодняшний день | the verb itself; «сейчас» |
| играет ключевую роль, является неотъемлемой частью | what exactly depends on it |
| стоит отметить, важно отметить, следует подчеркнуть, необходимо понимать | delete |
| давайте разберёмся, погрузимся в, разложим по полочкам | delete, or say what we are about to check |
| в современном мире, в эпоху ИИ, открывает новые горизонты, настоящий прорыв | delete |
| уникальный, инновационный, революционный, широкий спектр, многогранный | a fact or a number |
| не просто X, а Y; это не магия, это математика | say Y |
| ответ прост:, секрет в том, что, и вот тут начинается самое интересное, спойлер: | state the fact |
| в заключение, подводя итог, таким образом (as a section closer) | end on the last point |
| chains of «кроме того», «более того», «при этом» | split the sentence, or delete the joint |
| chains of «позволяет»: «что позволяет ..., что позволяет ...» | name who does what |
| nouns made from verbs: «проведение анализа», «осуществление обучения» | «анализируем», «обучаем» |

## What the editor never touches

- Facts, numbers, dates, names, quotes, citations and links. A number may be removed only together with its
  claim, never "rounded for style".
- Code, formulas, figure references, table contents and technical terms, including terms of art that look like
  jargon ("logit lens", "KV cache").
- The meaning of a claim, including its strength. Do not turn "suggests" into "shows" or the reverse.
- Deliberate roughness, jokes and asides that belong to the author, even when the editor would phrase them
  differently.
- In lessons and posts exported from the course repository (transformers-from-scratch), the site file. Edits go
  to the source (`architectures/aNN_*/lesson.py`, `lessons/NN_*/lesson.py`, `articles/<name>/`); the post is
  then exported again.

If a sentence is unclear and the editor cannot tell what it means, the editor asks. Never guess, and never invent
a fact to make a sentence concrete.

## How to edit

1. **Run the linter**: `python bin/style-lint.py <file>`. Its flags are candidates, not verdicts.
2. **Read the whole text once as a reader.** Mark where attention drops and where the text starts to sound like
   a brochure or a press release.
3. **Line edit.** Prefer deleting to rephrasing, and prefer a verb to a noun. Replace the general with the
   specific the author already gave elsewhere in the text. Don't add new material.
4. **The voice test.** Read the opening paragraph, then any paragraph from the middle. Do they sound like the same
   person? If a paragraph sounds like someone else (a textbook, a press release, a chatbot), rewrite it in the
   opening's voice.
5. **Titles last.** Title, description, headings, figure titles, image alt texts and cover text are checked
   against the structure rules above.
6. **Finish with the linter again.** Every remaining flag is either fixed or justified in the change log.

The result is the edited text plus a short change log: what was cut or rewritten and why, in a few lines, and
the questions the editor could not resolve.

## Before and after

Before: "In today's fast-paced world of AI, attention mechanisms play a crucial role. Let's dive into how they
work!"

After: "Every token in GPT-4o's answer was chosen after looking back at every token before it. That look back is
attention, and its cost grows with the square of the context."

Before: "The result? A whopping drop in accuracy. This isn't just a bug, it's a fundamental limitation."

After: "With a quarter of the image tokens kept, accuracy on receipts fell from 88% to 49%. A total printed once
on a receipt has no backup."

Before: "It's worth noting that the model, interestingly, seems to understand the concept of a receipt."

After: "The model reads the numbers and picks the wrong line: 100,000, the cash handed over, instead of the
total."

Before: «Данная модель является ключевым инструментом, который позволяет осуществлять анализ изображений.»

After: «Модель читает изображение как последовательность токенов, так же как текст.»

Before: «Стоит отметить, что результаты открывают новые горизонты. Ответ прост: всё дело в данных.»

After: «В смесь LLaVA-1.5 добавили 40 тысяч чисто текстовых диалогов, чтобы модель не разучилась просто
разговаривать.»

## Checklist

- [ ] The linter is clean, or every remaining flag is justified.
- [ ] No list in the body of a post; paragraphs of one to three sentences.
- [ ] The title, headings, figure titles, description and cover are plain clauses with no colon taglines, no
      "X, explained" and no slogans.
- [ ] No sentence opens with a staged reveal or answers its own rhetorical question.
- [ ] No paragraph ends on a moral.
- [ ] Every number, fact and quote is unchanged and still has its source.
- [ ] The opening paragraph and a random middle paragraph sound like the same person.
- [ ] Emphasis is italics only; at most one em dash per paragraph.
