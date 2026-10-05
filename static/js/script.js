/* =========================================================
   PostCraft AI — Frontend interactions
   ---------------------------------------------------------
   The Generate Posts button POSTs the idea, the selected
   platforms and the selected tone to the Flask backend
   (POST /api/generate) and renders the returned posts.

   Everything backend-facing lives behind one function —
   requestGeneration() — so the Gemini call on the server can
   change without touching the UI code.
   ========================================================= */
(function () {
  'use strict';

  /* ---------------------------------------------------------
     1. Configuration
     --------------------------------------------------------- */

  var MAX_CHARS = 5000;

  var API_ENDPOINT = '/api/generate';
  var REGENERATE_ENDPOINT = '/api/regenerate';

  var PLATFORMS = {
    linkedin: { id: 'linkedin', name: 'LinkedIn',        icon: '#i-linkedin', limit: 3000 },
    x:        { id: 'x',        name: 'X',               icon: '#i-x',        limit: 280  },
    medium:   { id: 'medium',   name: 'Dev.to / Medium', icon: '#i-medium',   limit: null }
  };

  var editorSeq = 0;

  /* Last successful submission — Regenerate replays it. */
  var lastRequest = null;

  /* ---------------------------------------------------------
     2. Helpers
     --------------------------------------------------------- */

  function $(selector, root) { return (root || document).querySelector(selector); }
  function $$(selector, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(selector));
  }

  function icon(id, className) {
    return '<svg class="icon' + (className ? ' ' + className : '') +
           '" aria-hidden="true"><use href="' + id + '"></use></svg>';
  }

  function countWords(text) {
    var trimmed = text.trim();
    return trimmed ? trimmed.split(/\s+/).length : 0;
  }

  function reducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  /* ---------------------------------------------------------
     3. DOM references
     --------------------------------------------------------- */

  var form        = $('#post-form');
  var ideaInput   = $('#idea');
  var counter     = $('#char-count');
  var errorBox    = $('#form-error');
  var generateBtn = $('#generate-btn');

  var resultsSection = $('#results');
  var resultsList    = $('#results-list');
  var resultsEmpty   = $('#results-empty');
  var resultsTitle   = $('#results-title');
  var resultsSub     = $('#results-sub');
  var copyAllBtn     = $('#copy-all');

  var nav        = $('#primary-nav');
  var navToggle  = $('#nav-toggle');
  var siteHeader = $('.site-header');
  var toTop      = $('#to-top');

  resultsTitle.setAttribute('tabindex', '-1');

  /* ---------------------------------------------------------
     4. Character counter
     --------------------------------------------------------- */

  function updateCounter() {
    var length = ideaInput.value.length;
    counter.textContent = length + ' / ' + MAX_CHARS;
    counter.classList.toggle('is-warning', length >= MAX_CHARS * 0.9 && length < MAX_CHARS);
    counter.classList.toggle('is-full', length >= MAX_CHARS);
  }

  ideaInput.addEventListener('input', function () {
    updateCounter();
    hideError();
  });

  /* ---------------------------------------------------------
     5. Platform + tone selection
     --------------------------------------------------------- */

  function syncGroup(selector, className) {
    $$(selector).forEach(function (item) {
      item.classList.toggle(className, $('input', item).checked);
    });
  }

  $$('.platform-card input').forEach(function (input) {
    input.addEventListener('change', function () {
      syncGroup('.platform-card', 'is-selected');
      hideError();
    });
  });

  $$('.tone-chip input').forEach(function (input) {
    input.addEventListener('change', function () {
      syncGroup('.tone-chip', 'is-selected');
    });
  });

  function selectedPlatforms() {
    return $$('.platform-card input:checked').map(function (input) { return input.value; });
  }

  function selectedTone() {
    var checked = $('input[name="tone"]:checked');
    return checked ? checked.value : 'Professional';
  }

  /* ---------------------------------------------------------
     6. Validation
     --------------------------------------------------------- */

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function hideError() {
    errorBox.hidden = true;
    errorBox.textContent = '';
  }

  function validate() {
    if (ideaInput.value.trim().length === 0) {
      showError('Tell us what you want to post about — rough notes are fine.');
      ideaInput.focus();
      return false;
    }
    if (selectedPlatforms().length === 0) {
      showError('Pick at least one platform to post to.');
      var first = $('.platform-card input');
      if (first) first.focus();
      return false;
    }
    hideError();
    return true;
  }

  /* ---------------------------------------------------------
     7. GENERATION SEAM  ← the only backend-facing function
     ---------------------------------------------------------
     POSTs { content, platforms, tone, variant } to the Flask API
     and resolves with the parsed JSON body:

       { success: true,
         tone_label: "Professional",
         results: [ { platform: "linkedin", content: "...", count: 120 }, ... ] }

     Rejects with an Error carrying a user-facing message.
     --------------------------------------------------------- */

  function ApiError(message, status) {
    this.name = 'ApiError';
    this.message = message;
    this.status = status;
  }
  ApiError.prototype = Object.create(Error.prototype);
  ApiError.prototype.constructor = ApiError;

  var GENERIC_ERROR = 'Something went wrong while generating your posts. Please try again.';
  var OFFLINE_ERROR = 'Could not reach the server. Make sure Flask is running, then try again.';

  function friendlyMessage(data, status) {
    if (data && typeof data.error === 'string' && data.error) return data.error;

    if (status === 404) return 'Generation endpoint not found. Is the backend up to date?';
    if (status === 405) return 'The server rejected that request type.';
    if (status === 413) return 'That idea is too long — keep it under ' + MAX_CHARS + ' characters.';
    if (status === 429) return 'Too many requests. Wait a moment and try again.';
    if (status >= 500) return 'The server had a problem generating your posts. Please try again.';
    return GENERIC_ERROR;
  }

  function postJson(url, payload) {
    return fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(payload)
    }).then(function (response) {
      return response.json()
        .catch(function () { return null; })
        .then(function (data) {
          if (!response.ok || !data || data.success !== true) {
            throw new ApiError(friendlyMessage(data, response.status), response.status);
          }
          return data;
        });
    }).catch(function (error) {
      if (error instanceof ApiError) throw error;
      /* fetch() itself only rejects on a network/CORS/abort failure. */
      throw new ApiError(OFFLINE_ERROR, 0);
    });
  }

  function requestGeneration(payload) {
    return postJson(API_ENDPOINT, payload).then(function (data) {
      if (!Array.isArray(data.results) || data.results.length === 0) {
        throw new ApiError('The server returned no posts. Please try again.', 0);
      }
      return data;
    });
  }

  /* Regenerates exactly one platform. The backend replies with a single
     `result` object instead of a `results` array. */
  function requestRegeneration(payload) {
    return postJson(REGENERATE_ENDPOINT, payload).then(function (data) {
      if (!data.result) {
        throw new ApiError('The server returned no post. Please try again.', 0);
      }
      return data;
    });
  }

  /* ---------------------------------------------------------
     8. Result cards
     --------------------------------------------------------- */

  function skeletonMarkup(rows) {
    var widths = ['96%', '88%', '72%', '94%', '62%', '84%'];
    var html = '';
    for (var i = 0; i < rows; i += 1) {
      html += '<span class="skeleton-line" style="width:' + widths[i % widths.length] + '"></span>';
    }
    return html;
  }

  /* state: 'ready' | 'loading' | 'error'
     opts:  { loading: bool, errorMessage: string, variant: number } */
  function buildCard(platformId, tone, opts) {
    opts = opts || {};

    var cfg = PLATFORMS[platformId];
    var loading = !!opts.loading;
    var failed = !!opts.errorMessage;
    var state = loading ? 'loading' : (failed ? 'error' : 'ready');

    editorSeq += 1;
    var editorId = 'post-editor-' + editorSeq;

    var card = document.createElement('article');
    card.className = 'result-card';
    card.dataset.platform = platformId;
    card.dataset.tone = tone;
    card.dataset.state = state;
    card.dataset.variant = String(
      typeof opts.variant === 'number' ? opts.variant : 0
    );

    card.innerHTML =
      '<header class="result-head">' +
        '<span class="result-id">' +
          '<span class="result-icon">' + icon(cfg.icon) + '</span>' +
          '<span class="result-id-text">' +
            '<span class="result-name">' + cfg.name + '</span>' +
            '<span class="result-sub">' + tone + ' tone</span>' +
          '</span>' +
        '</span>' +
        (cfg.limit ? '<span class="result-limit" data-limit></span>' : '') +
      '</header>' +

      '<div class="result-body">' +
        '<p class="result-text" data-text></p>' +
        '<p class="result-error" data-error-state hidden></p>' +
        '<div class="result-skeleton" data-skeleton hidden>' +
          '<div class="skeleton-stack">' + skeletonMarkup(cfg.limit === 280 ? 3 : 5) + '</div>' +
        '</div>' +
        '<div class="result-editor" data-editor hidden>' +
          '<label class="sr-only" for="' + editorId + '">Edit ' + cfg.name + ' post</label>' +
          '<textarea class="textarea" id="' + editorId + '" data-textarea rows="8" spellcheck="true"></textarea>' +
        '</div>' +
      '</div>' +

      '<footer class="result-foot">' +
        '<span class="result-count" data-count></span>' +
        '<p class="card-error" data-card-error hidden></p>' +
        '<div class="result-actions" data-actions="view">' +
          '<button class="btn btn-soft btn-sm" type="button" data-action="copy">' +
            icon('#i-copy') + '<span data-copy-label>Copy</span></button>' +
          '<button class="btn btn-soft btn-sm" type="button" data-action="edit">' +
            icon('#i-edit') + '<span>Edit</span></button>' +
          '<button class="btn btn-soft btn-sm" type="button" data-action="regenerate">' +
            icon('#i-refresh') + '<span data-regen-label>Regenerate</span></button>' +
        '</div>' +
        '<div class="result-actions" data-actions="edit" hidden>' +
          '<button class="btn btn-primary btn-sm" type="button" data-action="save">Save changes</button>' +
          '<button class="btn btn-soft btn-sm" type="button" data-action="cancel">Cancel</button>' +
        '</div>' +
      '</footer>';

    if (failed) {
      $('[data-text]', card).hidden = true;
      var box = $('[data-error-state]', card);
      /* textContent, never innerHTML: the message may come from the server. */
      box.textContent = opts.errorMessage;
      box.hidden = false;
      setRegenerateLabel(card, 'Retry');
      $('[data-count]', card).textContent = 'Not generated';
    }

    setCardLoading(card, loading, 'Writing…');
    return card;
  }

  function setCardText(card, text) {
    $('[data-text]', card).textContent = text;
    $('[data-textarea]', card).value = text;
    updateCount(card, text);
  }

  function updateCount(card, text) {
    var cfg = PLATFORMS[card.dataset.platform];
    var chars = text.length;
    var words = countWords(text);
    var countEl = $('[data-count]', card);
    var limitEl = $('[data-limit]', card);

    if (cfg.limit) {
      countEl.textContent = chars + ' / ' + cfg.limit + ' characters · ' + words + ' words';
      countEl.classList.toggle('is-over', chars > cfg.limit);
      limitEl.textContent = chars + ' / ' + cfg.limit;
    } else {
      countEl.textContent = chars + ' characters · ' + words + ' words';
      countEl.classList.remove('is-over');
    }
  }

  /* Puts a single card into (or out of) its loading state. A card that failed
     to generate keeps its error message instead of showing a skeleton. */
  function setCardLoading(card, loading, statusText) {
    var failed = card.dataset.state === 'error';

    if (!failed) $('[data-text]', card).hidden = loading;
    $('[data-skeleton]', card).hidden = !loading || failed;
    if (!loading) $('[data-editor]', card).hidden = true;

    $('[data-actions="view"]', card).hidden = loading;
    $('[data-actions="edit"]', card).hidden = true;

    var countEl = $('[data-count]', card);
    if (failed) {
      countEl.textContent = 'Not generated';
      countEl.classList.remove('is-over');
    } else if (loading) {
      countEl.textContent = statusText || 'Writing…';
      countEl.classList.remove('is-over');
    } else {
      updateCount(card, $('[data-text]', card).textContent);
    }
  }

  /* ---------------------------------------------------------
     8b. Card state helpers
     --------------------------------------------------------- */

  function setRegenerateLabel(card, label) {
    $('[data-regen-label]', card).textContent = label;
  }

  /* Only this card's Regenerate button changes; every other card stays usable. */
  function setRegenerating(card, on) {
    var button = $('[data-action="regenerate"]', card);
    button.disabled = on;
    button.classList.toggle('is-busy', on);
    if (on) {
      setRegenerateLabel(card, 'Regenerating…');
    } else {
      setRegenerateLabel(card, card.dataset.state === 'error' ? 'Retry' : 'Regenerate');
    }
  }

  function showCardError(card, message) {
    var box = $('[data-card-error]', card);
    box.textContent = message;
    box.hidden = false;
  }

  function hideCardError(card) {
    var box = $('[data-card-error]', card);
    box.hidden = true;
    box.textContent = '';
  }

  /* Puts new text into a card. Also revives a card that had failed. */
  function applyResult(card, result) {
    var wasFailed = card.dataset.state === 'error';

    if (wasFailed) {
      var box = $('[data-error-state]', card);
      box.hidden = true;
      box.textContent = '';
    }
    card.dataset.state = 'ready';
    setRegenerateLabel(card, 'Regenerate');

    $('[data-text]', card).hidden = false;
    setCardText(card, result.content);
    if (typeof result.variant === 'number') {
      card.dataset.variant = String(result.variant);
    }
  }

  /* ---------------------------------------------------------
     9. Clipboard
     --------------------------------------------------------- */

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var helper = document.createElement('textarea');
      helper.value = text;
      helper.setAttribute('readonly', '');
      helper.style.position = 'fixed';
      helper.style.top = '-1000px';
      document.body.appendChild(helper);
      helper.select();
      var ok = false;
      try { ok = document.execCommand('copy'); } catch (err) { ok = false; }
      document.body.removeChild(helper);
      ok ? resolve() : reject(new Error('copy failed'));
    });
  }

  function flashButton(button, labelEl, message, isError) {
    if (labelEl) labelEl.textContent = message;
    button.classList.toggle('is-done', !isError);
    button.classList.toggle('is-failed', !!isError);

    window.setTimeout(function () {
      if (labelEl) labelEl.textContent = button.dataset.action === 'copy' ? 'Copy'
        : button.dataset.action === 'regenerate' ? 'Regenerate' : 'Copy all';
      button.classList.remove('is-done', 'is-failed');
    }, 1700);
  }

  /* ---------------------------------------------------------
     10. Rendering results
     --------------------------------------------------------- */

  function renderResults(data, fallbackTone) {
    resultsList.innerHTML = '';
    var tone = data.tone_label || fallbackTone;
    var failed = 0;

    data.results.forEach(function (result) {
      var platformId = PLATFORMS[result.platform] ? result.platform : null;
      if (!platformId) return;

      if (result.success === false) {
        /* One platform failed; the rest of the cards still render. */
        failed += 1;
        resultsList.appendChild(buildCard(platformId, tone, {
          errorMessage: result.error || GENERIC_ERROR,
          variant: result.variant
        }));
        return;
      }

      var card = buildCard(platformId, tone, { variant: result.variant });
      setCardText(card, result.content);
      resultsList.appendChild(card);
    });

    var shown = resultsList.children.length;
    resultsEmpty.hidden = true;
    copyAllBtn.hidden = shown === 0;
    resultsList.setAttribute('aria-busy', 'false');

    var message = 'One idea, shaped for ' + shown +
      (shown === 1 ? ' platform.' : ' platforms.');
    if (failed > 0) {
      message += ' ' + failed + ' could not be generated — use Retry on those.';
    }
    resultsSub.textContent = message + ' Tweak anything before you post.';
  }

  function showSkeletons(platformIds, tone) {
    resultsList.innerHTML = '';
    resultsList.setAttribute('aria-busy', 'true');

    platformIds.forEach(function (id) {
      resultsList.appendChild(buildCard(id, tone, { loading: true }));
    });

    resultsEmpty.hidden = true;
    copyAllBtn.hidden = false;
  }

  function resetResults() {
    resultsList.innerHTML = '';
    resultsList.setAttribute('aria-busy', 'false');
    resultsEmpty.hidden = false;
    copyAllBtn.hidden = true;
  }

  function setFormLoading(loading) {
    generateBtn.classList.toggle('is-loading', loading);
    generateBtn.disabled = loading;
    generateBtn.setAttribute('aria-busy', loading ? 'true' : 'false');
  }

  /* ---------------------------------------------------------
     11. Submit
     --------------------------------------------------------- */

  form.addEventListener('submit', function (event) {
    event.preventDefault();

    if (generateBtn.disabled) return;
    if (!validate()) return;

    var content = ideaInput.value.trim();
    var platforms = selectedPlatforms();
    var tone = selectedTone();

    setFormLoading(true);
    showSkeletons(platforms, tone);

    requestGeneration({ content: content, platforms: platforms, tone: tone })
      .then(function (data) {
        lastRequest = { content: content, tone: data.tone_label || tone, platforms: platforms };
        renderResults(data, tone);
      })
      .catch(function (error) {
        resetResults();
        showError(error && error.message ? error.message : GENERIC_ERROR);
      })
      .then(function () {
        setFormLoading(false);
        resultsTitle.focus({ preventScroll: true });
        resultsSection.scrollIntoView({
          behavior: reducedMotion() ? 'auto' : 'smooth',
          block: 'start'
        });
      });
  });

  /* ---------------------------------------------------------
     12. Card actions (delegated)
     --------------------------------------------------------- */

  resultsList.addEventListener('click', function (event) {
    var button = event.target.closest('[data-action]');
    if (!button) return;

    var card = button.closest('.result-card');
    var action = button.getAttribute('data-action');
    var platformId = card.dataset.platform;

    if (action === 'copy') {
      copyText($('[data-text]', card).textContent).then(function () {
        flashButton(button, $('[data-copy-label]', button), 'Copied!');
      }).catch(function () {
        flashButton(button, $('[data-copy-label]', button), 'Copy failed', true);
      });
      return;
    }

    if (action === 'edit') { enterEditMode(card); return; }

    if (action === 'cancel') {
      exitEditMode(card, $('[data-textarea]', card).value);
      return;
    }

    if (action === 'save') {
      var next = $('[data-textarea]', card).value;
      if (!next.trim()) { $('[data-textarea]', card).focus(); return; }
      $('[data-text]', card).textContent = next;
      updateCount(card, next);
      exitEditMode(card, next);
      return;
    }

    /* Regenerates only the card that was clicked. Every other card is left
       exactly as it is, and the existing text survives a failed attempt. */
    if (action === 'regenerate') {
      if (!lastRequest || !lastRequest.content) {
        showError('Add your idea again to regenerate this post.');
        return;
      }

      var variant = (parseInt(card.dataset.variant, 10) || 0) + 1;
      setRegenerating(card, true);
      hideCardError(card);

      requestRegeneration({
        content: lastRequest.content,
        platform: platformId,
        tone: card.dataset.tone,
        variant: variant
      }).then(function (data) {
        var result = data.result;
        if (!result || result.success !== true) {
          throw new ApiError((result && result.error) || GENERIC_ERROR, 0);
        }
        applyResult(card, result);
      }).catch(function (error) {
        /* Keeps the previous text so the user does not lose a good post. */
        showCardError(card, error && error.message ? error.message : GENERIC_ERROR);
      }).then(function () {
        setRegenerating(card, false);
      });
    }
  });

  function enterEditMode(card) {
    var textarea = $('[data-textarea]', card);
    textarea.value = $('[data-text]', card).textContent;

    $('[data-text]', card).hidden = true;
    $('[data-skeleton]', card).hidden = true;
    $('[data-editor]', card).hidden = false;
    $('[data-actions="view"]', card).hidden = true;
    $('[data-actions="edit"]', card).hidden = false;

    updateCount(card, textarea.value);
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
  }

  function exitEditMode(card, text) {
    if (typeof text === 'string') $('[data-text]', card).textContent = text;
    $('[data-text]', card).hidden = false;
    $('[data-editor]', card).hidden = true;
    $('[data-actions="view"]', card).hidden = false;
    $('[data-actions="edit"]', card).hidden = true;
    updateCount(card, $('[data-text]', card).textContent);
  }

  /* Live counts while editing */
  resultsList.addEventListener('input', function (event) {
    if (!event.target.matches('[data-textarea]')) return;
    var card = event.target.closest('.result-card');
    updateCount(card, event.target.value);
  });

  /* Ctrl/Cmd + Enter saves from inside the editor */
  resultsList.addEventListener('keydown', function (event) {
    if (!event.target.matches('[data-textarea]')) return;
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault();
      var saveBtn = event.target.closest('.result-card').querySelector('[data-action="save"]');
      if (saveBtn) saveBtn.click();
    }
  });

  /* ---------------------------------------------------------
     13. Copy all
     --------------------------------------------------------- */

  copyAllBtn.addEventListener('click', function () {
    var label = $('span', copyAllBtn);
    /* Only cards that actually hold a post. */
    var blocks = $$('.result-card', resultsList)
      .filter(function (card) { return card.dataset.state === 'ready'; })
      .map(function (card) {
        var cfg = PLATFORMS[card.dataset.platform];
        var rule = '';
        for (var i = 0; i < cfg.name.length + 4; i += 1) rule += '-';
        return cfg.name + '\n' + rule + '\n' + $('[data-text]', card).textContent;
      });

    if (!blocks.length) return;

    copyText(blocks.join('\n\n\n')).then(function () {
      flashButton(copyAllBtn, label, 'Copied all!');
    }).catch(function () {
      flashButton(copyAllBtn, label, 'Copy failed', true);
    });
  });

  /* ---------------------------------------------------------
     14. Navigation, sticky header, back-to-top
     --------------------------------------------------------- */

  function closeNav() {
    if (!nav.classList.contains('is-open')) return;
    nav.classList.remove('is-open');
    navToggle.setAttribute('aria-expanded', 'false');
    navToggle.setAttribute('aria-label', 'Open menu');
  }

  navToggle.addEventListener('click', function () {
    var open = nav.classList.toggle('is-open');
    navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    navToggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
  });

  document.addEventListener('click', function (event) {
    if (event.target.closest('a[href^="#"]')) closeNav();
    else if (!event.target.closest('.site-header')) closeNav();
  });

  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && nav.classList.contains('is-open')) {
      closeNav();
      navToggle.focus();
    }
  });

  /* Header shadow + back-to-top */
  var lastScrollY = -1;
  function onScroll() {
    var y = window.pageYOffset || document.documentElement.scrollTop;
    if (y === lastScrollY) return;
    lastScrollY = y;

    siteHeader.classList.toggle('is-stuck', y > 8);

    var show = y > 600;
    toTop.hidden = !show;
    toTop.classList.toggle('is-visible', show);
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  toTop.addEventListener('click', function () {
    window.scrollTo({ top: 0, behavior: reducedMotion() ? 'auto' : 'smooth' });
  });

  /* Active nav link for the section in view */
  if ('IntersectionObserver' in window) {
    var navLinks = $$('.nav-link');
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        navLinks.forEach(function (link) {
          link.classList.toggle('is-active', link.getAttribute('href') === '#' + entry.target.id);
        });
      });
    }, { rootMargin: '-45% 0px -50% 0px', threshold: 0 });

    ['create', 'how-it-works', 'about'].forEach(function (id) {
      var el = document.getElementById(id);
      if (el) observer.observe(el);
    });
  }

  /* ---------------------------------------------------------
     15. Init
     --------------------------------------------------------- */

  updateCounter();
  syncGroup('.platform-card', 'is-selected');
  syncGroup('.tone-chip', 'is-selected');
})();