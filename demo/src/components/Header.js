import React from 'react';

/*******************************************************************************
  <Header /> Component
*******************************************************************************/

class Header extends React.Component {
    render() {
      const { selectedModel, changeModel, permalink } = this.props;

      const buildLink = (thisModel, label) => {
        return (
          <li>
            <a href="#" className={`nav__link ${selectedModel === thisModel ? "nav__link--selected" : ""}`} onClick={() => { changeModel(thisModel) }}>
              <span>{label}</span>
            </a>
          </li>
        )
      }

      // Header links are different in the permalink and non-permalink case.
      let links = null;

      if (permalink) {
        // This is the header for a permalink page, so we grab the root URL
        // and just provide a link to it.
        const slugRegex = /\/permalink\/([^/]+)\/?$/;
        const demoRoot = window.location.href.replace(slugRegex, '');

        links = (
          <ul>
            <li>
              <a href={demoRoot} className="nav__link" target="_blank">
                <span>Try Your Own</span>
              </a>
            </li>
          </ul>
        )
      } else {
        links = (
          <ul>
            {buildLink("semantic-role-labeling", "SRL Model")}
            {buildLink("machine-comprehension", "MC Model")}
            {buildLink("textual-entailment", "TE Model")}
          </ul>
        )
      }

      return (
        <header>
          <div className="header__content">
            <nav>

              <ul>
              {links}
              </ul>
            </nav>
            <h1 className="header__content__logo">
              <a href="http://www.allennlp.org/" target="_blank" rel="noopener noreferrer">
                <svg>
                  <use xlinkHref="#icon__allennlp-logo"></use>
                </svg>
                <span className="u-hidden">AllenNLP</span>
              </a>
            </h1>
          </div>
        </header>
      );
    }
  }

export default Header;
