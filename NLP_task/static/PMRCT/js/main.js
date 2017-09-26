$(document).ready(function() {

    // AJAX
    $('#submit_phrases').click(function(){
        var phrase_confirmed = [];
        var phrase_dismissed = [];

        $(".phrase_confirm").each(function(){
            if($(this).prop("checked"))  phrase_confirmed.push($(this).val());
            if (!$(this).prop("checked"))  phrase_dismissed.push($(this).val());
        });

        phrase_confirmed = {
            'phrase': phrase_confirmed.join('||'),
            'phrase_no': phrase_dismissed.join('||')
        };


        $.ajax({
            type: "POST",
            url: "/PMRCT/register_new_phrases/",
            dataType: "json",
            data: phrase_confirmed,
            success: function(data) {
                location.reload();
            }
        });

    });


    $('#submit_words').click(function(){
        var word_confirmed = {};
        var behavior_confirmed = [];
        var behavior_dismissed = [];
        var change_confirmed = [];
        var change_dismissed = [];
        var middle_confirmed = [];
        var middle_dismissed = [];

        $(".behavior_confirm").each(function(){
            if($(this).prop("checked"))  behavior_confirmed.push($(this).val());
            if (!$(this).prop("checked"))  behavior_dismissed.push($(this).val());
        });

        $(".change_confirm").each(function(){
            if($(this).prop("checked"))  change_confirmed.push($(this).val());
            if(!$(this).prop("checked"))  change_dismissed.push($(this).val());
        });

        $(".middle_confirm").each(function(){
            if($(this).prop("checked"))  middle_confirmed.push($(this).val());
            if(!$(this).prop("checked"))  middle_dismissed.push($(this).val());
        });

        word_confirmed = {
            'behavior': behavior_confirmed.join('||'),
            'behavior_no': behavior_dismissed.join('||'),
            'change': change_confirmed.join('||'),
            'change_no': change_dismissed.join('||'),
            'middle': middle_confirmed.join('||'),
            'middle_no': middle_dismissed.join('||')
        };

        // alert(JSON.stringify(word_confirmed));
        // alert('kuru');
        $.ajax({
            type: "POST",
            url: "/PMRCT/register_new_words/",
            dataType: "json",
            data: word_confirmed,
            success: function(data) {
                // $("#new_phrases").css("display", "none");
                active_learning();
            }
        });

    });

    function active_learning(){
        // alert('active_learning');
        $.ajax({
            type: "POST",
            url: "/PMRCT/active_learning/",
            dataType: "json",
            data: 'data_from_browser',
            success: function(data) {
                $("#new_words").css("display", "none");
                // alert(data.nearzeros);
                var nz = data.nearzeros;
                nearzeros = nz.split('||');

                console.log('1')

                $("#select_new_phrases").text('Select evidence phrases:');

                console.log('2')

                var nearzero_check = "<form><div>";
                for(i=0; i<nearzeros.length; i++){
                    nearzero_check = nearzero_check 
                    + '<div class="each_phrase"><input type="checkbox" class="nearzero_check" value="'  
                    + nearzeros[i] + '" checked="checked">' + nearzeros[i] + '</div>'
                }
                nearzero_check = nearzero_check + '</div></form>';
                $('#new_phrase_list').html(nearzero_check);
                $('.each_phrase').css('margin-top', '10px');
                $('#each_phrase').css('margin-bottom', '10px');
                $('#new_phrases').css('margin-top', '20px');
                $('#new_phrases').css('margin-bottom', '70px');
                $('#new_phrases').css('background-color', '#d6eff5');
                $('#new_phrases').css('padding-left', '20px');
                $('#new_phrases').css('padding-top', '20px');
                $('#new_phrases').css('padding-bottom', '40px');
                $('#submit_phrases').css('display','block');
                $('#submit_phrases').css('position','relative');
                $('#submit_phrases').css('left', '20px');
                $('#submit_phrases').css('top', '30px');
                $('#new_phrases').css('padding-bottom', '40px');   
                 console.log('3')            
            }
        });
    }


    $('#input_seed_data').click(function(){
        $.ajax({
            type: "POST",
            url: "/PMRCT/input_seed_data/",
            dataType: "json",
            data: { "item": 'data_from_browser'},
            success: function(data) {
                // alert(data.message);
            }
        });

    });

    // CSRF code
    function getCookie(name) {
        var cookieValue = null;
        var i = 0;
        if (document.cookie && document.cookie !== '') {
            var cookies = document.cookie.split(';');
            for (i; i < cookies.length; i++) {
                var cookie = jQuery.trim(cookies[i]);
                // Does this cookie string begin with the name we want?
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    var csrftoken = getCookie('csrftoken');

    function csrfSafeMethod(method) {
        // these HTTP methods do not require CSRF protection
        return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
    }
    $.ajaxSetup({
        crossDomain: false, // obviates need for sameOrigin test
        beforeSend: function(xhr, settings) {
            if (!csrfSafeMethod(settings.type)) {
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
            }
        }
    }); 


    $.ajax({
        type: "POST",
        url: "/PMRCT/search_new_snippets/",
        dataType: "json",
        data: { "item": 'data_from_browser'},
        success: function(data) {
            $("#progress_info1").html("Number of phrases collected: <span style='color:rgb(51, 51, 51)'>" + data.snippets_num + "</span>")
            $("#progress_info2").html("Applying SVMs classification to the phrases..")
            snippets_classification();
        }
    });

    function snippets_classification(){
        $.ajax({
            type: "POST",
            url: "/PMRCT/snippets_classification/",
            dataType: "json",
            data: { "item": 'data_from_browser'},
            success: function(data) {
                $("#progress_info2").html("SVMs classification has been completed.");
                $("#positive_result").html("Positive: <span>" + data.positive_num + "</span>");
                $("#negative_result").html("Negative: <span>" + data.negative_num + "</span>");
                $("#unknown_result").html("Unknown: <span>" + data.unknown_num + "</span>");

                var new_behavior = data.new_behavior;
                var new_change = data.new_change;
                var new_middle = data.new_middle;

                var behavior_check = "<form><p>";
                for(i=0; i<new_behavior.length; i++){
                    behavior_check = behavior_check 
                    + '<input type="checkbox" class="behavior_confirm" value="'  
                    + new_behavior[i] + '" checked="checked">' + new_behavior[i] + '&emsp;'
                }
                behavior_check = behavior_check + '</p></form>'

                var change_check = "<form><p>";
                for(i=0; i<new_change.length; i++){
                    change_check = change_check 
                    + '<input type="checkbox" class="change_confirm" value="'  
                    + new_change[i] + '" checked="checked">' + new_change[i] + '&emsp;'
                }
                change_check = change_check + '</p></form>'


                // var middle_check = "<form><p>";
                // for(i=0; i<new_middle.length; i++){
                //     middle_check = middle_check 
                //     + '<input type="checkbox" class="middle_confirm" value="'  
                //     + new_middle[i] + '" checked="checked">' + new_middle[i] + '&emsp;'
                // }
                // middle_check = middle_check + '</p></form>'

                if (new_behavior.length > 0)
                    $("#new_behavior").html('Select adequate "behavior": ' + behavior_check);

                if (new_change.length > 0)
                    $("#new_change").html('Select adequate  "change": ' + change_check);
    
                // if (new_middle.length > 0)
                //     $("#new_middle").html('Select adequate  "middle": ' + middle_check);

                $('#new_words').css('margin-top', '20px');
                $('#new_words').css('margin-bottom', '70px');
                $('#new_words').css('background-color', '#d6eff5');
                $('#new_words').css('padding-left', '20px');
                $('#new_words').css('padding-top', '20px');
                $('#new_change').css('margin-top', '-20px');
                // $('#new_middle').css('margin-top', '-20px');
                $('#submit_words').css('display','block');
                $('#submit_words').css('position','relative');
                $('#submit_words').css('left', '20px');
                $('#submit_words').css('top', '10px');
                $('#new_words').css('padding-bottom', '40px');
            }
        });        
    }

});