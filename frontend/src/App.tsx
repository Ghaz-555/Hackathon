function App() {
  return (
    <main>
      <h1>GlassBox</h1>
      <p>See inside a language model.</p>

      <section>
        <h2>Temperature Experiment</h2>
        <p>
          Change the temperature and watch how the model's next-character
          predictions change.
        </p>
      </section>

      <section>
        <h2>Training Lab</h2>
        <p>
          Train the model and compare training loss with held-out loss.
        </p>
      </section>

      <section>
        <h2>Attention</h2>
        <p>
          Explore which earlier characters the model pays attention to.
        </p>
      </section>

      <section>
        <h2>Generate</h2>
        <p>
          Generate text using temperature and a repeatable sampling seed.
        </p>
      </section>
    </main>
  )
}

export default App